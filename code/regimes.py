"""
Causal 2-state Gaussian HMM on realised volatility.

The parameters are estimated on the fit window only (2000-2015, configs/labelling.yaml)
and then frozen. Regime probabilities for every day come from an explicit forward
recursion with those parameters, so the probability for day t uses observations up
to t (filtered) or up to t-1 (one-step predicted), never later ones. hmmlearn's
predict and predict_proba are not used: both condition on the whole sequence.
"""

import numpy as np
import pandas as pd


def forward_filter(obs, params):
    """
    Forward recursion for a Gaussian HMM with one-dimensional observations.

    Returns (filtered, predicted, loglik): filtered[t] = P(s_t | y_1..y_t),
    predicted[t] = P(s_t | y_1..y_{t-1}) (the start distribution on the first
    observed day), and the log-likelihood of the observed values. Days before the
    first observation are NaN. A missing value later carries the prediction
    forward without an emission update.
    """
    y = obs.to_numpy(float)
    pi, A = params["startprob"], params["transmat"]
    mu, var = params["means"], params["vars"]
    K = len(mu)
    filt = np.full((len(y), K), np.nan)
    pred = np.full((len(y), K), np.nan)
    prev, loglik = None, 0.0
    for t in range(len(y)):
        if prev is None and np.isnan(y[t]):
            continue
        p = pi if prev is None else prev @ A
        pred[t] = p
        if np.isnan(y[t]):
            f = p
        else:
            logb = -0.5 * (np.log(2 * np.pi * var) + (y[t] - mu) ** 2 / var)
            m = logb.max()
            w = p * np.exp(logb - m)        # scaled to avoid underflow
            c = w.sum()
            loglik += m + np.log(c)
            f = w / c
        filt[t] = f
        prev = f
    cols = [f"p_state{k}" for k in range(K)]
    return (pd.DataFrame(filt, index=obs.index, columns=cols),
            pd.DataFrame(pred, index=obs.index, columns=cols), loglik)


def most_probable(probs):
    """Most probable state per day (P(high-vol) > 0.5 for two states); NaN where unknown."""
    regime = pd.Series(np.nan, index=probs.index, name="regime")
    known = probs.notna().all(axis=1)
    regime[known] = probs[known].to_numpy().argmax(axis=1)
    return regime


def fit_hmm(obs, fit_window, n_states, covariance_type, n_iter, random_state):
    """
    Fit a Gaussian HMM on the observations inside `fit_window` only. Returns the
    parameters with states ordered by mean (state 0 = low volatility), plus fit
    diagnostics. Fails if the forward recursion does not reproduce hmmlearn's
    log-likelihood on the fit window.
    """
    from hmmlearn.hmm import GaussianHMM

    train = obs.loc[pd.Timestamp(fit_window[0]):pd.Timestamp(fit_window[1])].dropna()
    X = train.to_numpy(float).reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                        n_iter=n_iter, random_state=random_state)
    model.fit(X)
    order = np.argsort(model.means_.ravel())
    params = {
        "startprob": model.startprob_[order],
        "transmat": model.transmat_[np.ix_(order, order)],
        "means": model.means_.ravel()[order],
        "vars": np.asarray(model.covars_).reshape(n_states, -1)[:, 0][order],
    }
    info = {
        "fit_start": str(train.index.min().date()), "fit_end": str(train.index.max().date()),
        "fit_days": len(train), "converged": bool(model.monitor_.converged),
        "iterations": int(model.monitor_.iter), "loglik_hmmlearn": float(model.score(X)),
    }
    _, _, loglik = forward_filter(train, params)
    info["loglik_forward_recursion"] = float(loglik)
    assert np.isclose(loglik, info["loglik_hmmlearn"], rtol=1e-8, atol=1e-6), \
        "forward recursion does not reproduce hmmlearn's log-likelihood"
    return params, info


def check_filter_causality(obs, params, n_checks=3, seed=0):
    """
    Replace every observation after day T with other values: the filtered
    probability at every day up to T, and the prediction for T+1, must be
    unchanged. Later probabilities must change, so the test is not vacuous.
    """
    filt, pred, _ = forward_filter(obs, params)
    rng = np.random.default_rng(seed)
    values = obs.dropna().to_numpy()
    n = len(obs)
    for q in np.linspace(0.3, 0.85, n_checks):
        pos = int(n * q)
        T, T_next = obs.index[pos], obs.index[pos + 1]
        after = obs.index > T
        pert = obs.copy()
        pert.loc[after] = rng.choice(values, size=int(after.sum()))
        f2, p2, _ = forward_filter(pert, params)
        np.testing.assert_array_equal(f2.loc[:T].to_numpy(), filt.loc[:T].to_numpy(),
                                      err_msg=f"filtered probabilities up to {T.date()} changed")
        np.testing.assert_array_equal(p2.loc[:T_next].to_numpy(), pred.loc[:T_next].to_numpy(),
                                      err_msg=f"predicted probabilities up to {T_next.date()} changed")
        assert not np.allclose(f2.loc[after].to_numpy(), filt.loc[after].to_numpy(), equal_nan=True), \
            f"causality test vacuous at {T.date()}"
    print(f"Causality check passed at {n_checks} dates: probabilities up to T are unchanged "
          "when every observation after T is replaced.")
