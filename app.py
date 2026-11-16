
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import cvxpy as cp
import warnings

# ============================================================
# PORTFOLIO OPTIMIZATION DASHBOARD
# Faithful Streamlit implementation of the original Python model
# Manual inputs - no Excel dependency
# ============================================================

# %% SETTINGS

st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📊",
    layout="wide"
)

DEFAULT_FRONTIER_POINTS = 100
SIMULATION = 50
ANNUALIZATION_FACTOR = 1
GAMMA_MIN = -3
GAMMA_MAX = 3
LAMBDA_DURATION = 1.0
SIM = 100

# Original PortfolioConstraints
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0
TARGET_SUM = 1.0

# Original reallocation settings
DEFAULT_SHIFT = 0.01

# Fixed model parameters (not exposed in Streamlit sidebar)

n_simulations = SIM
n_portfolios = DEFAULT_FRONTIER_POINTS

gamma_min = GAMMA_MIN
gamma_max = GAMMA_MAX

lambda_duration = LAMBDA_DURATION

use_seed = False
seed = 42

# %% PLOT SETTINGS

plt.rcParams.update({
    "font.family": "Century Gothic",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# %% DEFAULT INPUTS

DEFAULT_ASSETS = pd.DataFrame({
    "Asset": [
        "Corporate Hybrids",
        "Financials Capital Structure (Snr)",
        "Financials Capital Structure (Snr NP)",
        "Financials Capital Structure (LT2)",
        "Financials Capital Structure (AT1)",
        "X-Over BBB-BB",
        "Leverage Loans",
        "Structured Credit",
        "CLOs",
        "Equity High Dividend",
    ],
    "Expected Return (%)": [
        4.9, 6.0, 5.5, 7.1, 4.8,
        3.89, 4.5, 3.7, 5.2, 10.0
    ],
    "": [
        16.0, 30.2, 18.495, 26.167, 21.0,
        11.0, 46.0, 22.78, 22.0, 22.0
    ],
    "Duration": [
        4.0, 4.5, 4.5, 5.0, 5.0,
        3.0, 2.0, 3.0, 2.5, 7.0
    ]
})

# %% SUPPORT FUNCTIONS

def corr_to_cov(corr_matrix, stdevs):
    """
    Original correlation-to-covariance conversion.
    """
    return corr_matrix * np.outer(stdevs, stdevs)


def portfolio_risk(weights, covariance):
    """
    Portfolio standard deviation.
    """
    return float(np.sqrt(max(weights @ covariance @ weights, 0.0)))


def calculate_marginal_scr(weights, vol, corr, risk):
    """
    Faithful implementation of the original marginal SCR function.

    weights: DataFrame (portfolios x assets)
    vol: Series
    corr: DataFrame
    risk: array of portfolio risks
    """
    D = vol.values
    CORR = corr.values

    mSCR = np.zeros_like(weights.values)

    for p in range(len(weights)):
        W = weights.iloc[p].values
        SCR_COMPONENT = W * D

        if risk[p] > 1e-12:
            mSCR[p, :] = (
                D * (CORR @ SCR_COMPONENT) / risk[p]
            )

    return pd.DataFrame(
        mSCR,
        index=weights.index,
        columns=weights.columns
    )


def calculate_return_per_marginal_scr(weights, er, mSCR):
    """
    Original Return / Marginal SCR calculation.
    """
    n_port = len(weights)
    n_asset = len(weights.columns)

    der_rend = np.tile(
        er.loc[weights.columns].values,
        (n_port, 1)
    )

    rend_per_mscr = np.full(
        (n_port, n_asset),
        np.nan
    )

    for p in range(n_port):
        for i in range(n_asset):

            if abs(mSCR.iloc[p, i]) > 1e-12:
                rend_per_mscr[p, i] = (
                    der_rend[p, i] / mSCR.iloc[p, i]
                )

    return pd.DataFrame(
        rend_per_mscr,
        index=weights.index,
        columns=weights.columns
    )


def classify_allocation(dR, dS, tol=1e-12):

    if dR > tol and dS < -tol:
        return (
            "Aumento Rendimento / Diminuzione Rischio",
            "Miglioramento"
        )

    elif dR > tol and dS > tol:
        return (
            "Aumento Rendimento / Aumento Rischio",
            "Trade-off"
        )

    elif dR < -tol and dS < -tol:
        return (
            "Diminuzione Rendimento / Diminuzione Rischio",
            "Trade-off"
        )

    elif dR < -tol and dS > tol:
        return (
            "Diminuzione Rendimento / Aumento Rischio",
            "Peggioramento"
        )

    else:
        return (
            "Effetto nullo o invariato",
            "Da verificare"
        )


# %% OPTIMIZATION ENGINE

def optimize_frontier(
    simulated_returns,
    durations,
    duration_liab,
    liability_perc,
    asset_perc,
    n_portfolios,
    gamma_min,
    gamma_max,
    lambda_duration
):
    """
    Direct implementation of the original EfficientFrontier class.
    """

    # Original EfficientFrontier input calculations
    mu = simulated_returns.mean() * ANNUALIZATION_FACTOR

    covariance = (
        simulated_returns.cov() * ANNUALIZATION_FACTOR
    )

    duration = durations.loc[simulated_returns.columns].values

    duration_target = (
        duration_liab * liability_perc / asset_perc
    )

    n_assets = len(simulated_returns.columns)

    # CVXPY variables
    w = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)

    # Original objective
    risk = cp.quad_form(
        w,
        covariance.values
    )

    ret = mu.values.T @ w

    duration_asset = duration @ w

    duration_mismatch = (
        duration_asset - duration_target
    )

    objective = cp.Maximize(
        ret
        - gamma * risk
        - lambda_duration * cp.square(duration_mismatch)
    )

    # Original PortfolioConstraints
    constraints = [
        cp.sum(w) == TARGET_SUM,
        w >= min_weight,
        w <= max_weight
    ]

    problem = cp.Problem(
        objective,
        constraints
    )

    gamma_values = np.logspace(
        gamma_min,
        gamma_max,
        num=n_portfolios
    )

    all_weights = []
    all_returns = []
    all_risks = []

    for gamma_value in gamma_values:

        gamma.value = float(gamma_value)

        # IMPORTANT:
        # Same default solver selection as original code.
        problem.solve()

        if (
            problem.status not in
            (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            or w.value is None
        ):
            raise RuntimeError(
                f"Optimization failed for gamma={gamma_value:.6g}. "
                f"Solver status: {problem.status}"
            )

        weights = np.asarray(w.value).flatten()

        portfolio_return = float(ret.value)

        portfolio_risk_value = float(
            np.sqrt(max(risk.value, 0.0))
        )

        all_weights.append(weights)
        all_returns.append(portfolio_return)
        all_risks.append(portfolio_risk_value)

    return (
        pd.DataFrame(
            all_weights,
            columns=simulated_returns.columns
        ),
        np.asarray(all_returns),
        np.asarray(all_risks)
    )


# %% MONTE CARLO SIMULATION

def simulate(
    er,
    cov,
    n_simulations,
    n_observations,
    durations,
    duration_liab,
    liability_perc,
    asset_perc,
    n_portfolios,
    gamma_min,
    gamma_max,
    lambda_duration,
    seed=None,
    progress_callback=None
):
    """
    Faithful implementation of the original simulate() function.

    For every simulation:
    - Generate 50 multivariate normal observations.
    - Calculate sample expected returns and covariance.
    - Optimize 100 portfolios.
    """

    if seed is not None:
        np.random.seed(int(seed))

    weights_simulations = []
    risk_simulations = []
    return_simulations = []

    dates = pd.date_range(
        start="2026-09-30",
        periods=n_observations,
        freq="YE"
    )

    for sim in range(n_simulations):

        simulation_data = pd.DataFrame(
            np.random.multivariate_normal(
                mean=er.values,
                cov=cov.values,
                size=len(dates)
            ),
            columns=cov.columns,
            index=dates
        )

        weights, returns, risks = optimize_frontier(
            simulated_returns=simulation_data,
            durations=durations,
            duration_liab=duration_liab,
            liability_perc=liability_perc,
            asset_perc=asset_perc,
            n_portfolios=n_portfolios,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            lambda_duration=lambda_duration
        )

        weights_simulations.append(weights)
        risk_simulations.append(risks)
        return_simulations.append(returns)

        if progress_callback is not None:
            progress_callback(sim + 1, n_simulations)

    return (
        weights_simulations,
        risk_simulations,
        return_simulations
    )


# %% ORIGINAL MONTE CARLO AGGREGATION

def aggregate_simulations(
    weights_simulations,
    risk_simulations,
    return_simulations,
    er,
    durations,
    duration_liab,
    liability_perc,
    asset_perc
):
    """
    Same aggregation logic as the original Python code.
    """

    avg_risk = pd.DataFrame(
        risk_simulations
    ).mean()

    avg_ret = pd.DataFrame(
        return_simulations
    ).mean()

    avg_weights = (
        pd.concat(weights_simulations)
        .groupby(level=0)
        .mean()
    )

    exp_returns = (
        avg_weights @ er
    ) * ANNUALIZATION_FACTOR

    duration_asset = (
        avg_weights @ durations.loc[avg_weights.columns]
    )

    duration_target = float(
        duration_liab * liability_perc / asset_perc
    )

    duration_mismatch = pd.Series(
        duration_asset.values - duration_target,
        index=duration_asset.index
    )

    opt = pd.concat(
        [
            avg_weights,
            avg_risk.rename("Risk"),
            avg_ret.rename("Exp_Ret_Sim"),
            exp_returns.rename("Exp_Ret")
        ],
        axis=1
    )

    opt["Duration_Asset"] = duration_asset
    opt["Duration_Liability"] = duration_liab
    opt["Duration_Mismatch"] = duration_mismatch

    return (
        avg_weights,
        avg_risk,
        avg_ret,
        exp_returns,
        duration_asset,
        duration_mismatch,
        duration_target,
        opt
    )


# %% MARGINAL SCR MONTE CARLO

def calculate_marginal_averages(
    weights_simulations,
    risk_simulations,
    vol,
    corr,
    er
):
    """
    IMPORTANT:
    Reproduces the exact sequence of the original code.

    In particular, the cumulative average mSCR is updated
    before calculating Return / mSCR at each simulation.
    """

    mSCR_simulations = []
    return_per_mscr_simulations = []

    for p in range(len(weights_simulations)):

        # Marginal SCR for current simulation
        mSCR_current = calculate_marginal_scr(
            weights_simulations[p],
            vol.loc[weights_simulations[p].columns],
            corr.loc[
                weights_simulations[p].columns,
                weights_simulations[p].columns
            ],
            risk_simulations[p]
        )

        mSCR_simulations.append(mSCR_current)

        # Original progressive average
        mSCR_average = (
            pd.concat(mSCR_simulations)
            .groupby(level=0)
            .mean()
        )

        # Original Return / mSCR
        return_per_mscr_current = (
            calculate_return_per_marginal_scr(
                weights_simulations[p],
                er,
                mSCR_average
            )
        )

        return_per_mscr_simulations.append(
            return_per_mscr_current
        )

    # Final average across simulations
    avg_mSCR = (
        pd.concat(mSCR_simulations)
        .groupby(level=0)
        .mean()
    )

    avg_return_per_mscr = (
        pd.concat(return_per_mscr_simulations)
        .groupby(level=0)
        .mean()
    )

    return avg_mSCR, avg_return_per_mscr


# %% REALLOCATION ANALYSIS

def calculate_reallocation_analysis(
    W0,
    asset_names,
    EXP_RET,
    COV,
    mSCR,
    shift=0.01,
    max_weight=1.0,
    min_weight=0.0
):

    n_asset = len(asset_names)

    DELTA_REND = (
        EXP_RET[:, None] - EXP_RET[None, :]
    )

    DELTA_RISK_MARG = (
        mSCR[:, None] - mSCR[None, :]
    )

    OPPORTUNITA_MARG = pd.DataFrame(
        "",
        index=asset_names,
        columns=asset_names
    )

    TRADEOFF_MARG = pd.DataFrame(
        "",
        index=asset_names,
        columns=asset_names
    )

    # Marginal opportunity and trade-off matrices
    for i in range(n_asset):

        for j in range(n_asset):

            if i == j:
                OPPORTUNITA_MARG.iloc[i, j] = "N/A"
                TRADEOFF_MARG.iloc[i, j] = "N/A"
                continue

            dR = DELTA_REND[i, j]
            dS = DELTA_RISK_MARG[i, j]

            classification, tradeoff = classify_allocation(
                dR,
                dS
            )

            OPPORTUNITA_MARG.iloc[i, j] = classification
            TRADEOFF_MARG.iloc[i, j] = tradeoff

    DELTA_REND_MARG = pd.DataFrame(
        DELTA_REND,
        index=asset_names,
        columns=asset_names
    )

    DELTA_RISK_MARG = pd.DataFrame(
        DELTA_RISK_MARG,
        index=asset_names,
        columns=asset_names
    )

    # Base portfolio
    REND_BASE = W0 @ EXP_RET
    RISK_BASE = np.sqrt(W0 @ COV @ W0)

    results = []

    # One-for-one reallocation
    for i in range(n_asset):

        for j in range(n_asset):

            if i == j:
                continue

            asset_increased = asset_names[i]
            asset_reduced = asset_names[j]

            if W0[j] - shift < min_weight - 1e-12:

                results.append({
                    "AssetAumentato": asset_increased,
                    "AssetRidotto": asset_reduced,
                    "DeltaRendimento": np.nan,
                    "DeltaRischio": np.nan,
                    "RendimentoNuovo": np.nan,
                    "RischioNuovo": np.nan,
                    "Classificazione": "Non fattibile"
                })

                continue

            if W0[i] + shift > max_weight + 1e-12:

                results.append({
                    "AssetAumentato": asset_increased,
                    "AssetRidotto": asset_reduced,
                    "DeltaRendimento": np.nan,
                    "DeltaRischio": np.nan,
                    "RendimentoNuovo": np.nan,
                    "RischioNuovo": np.nan,
                    "Classificazione": "Non fattibile"
                })

                continue

            W1 = W0.copy()

            W1[i] += shift
            W1[j] -= shift

            REND_NUOVO = W1 @ EXP_RET

            RISK_NUOVO = np.sqrt(
                W1 @ COV @ W1
            )

            DELTA_REND_EFF = (
                REND_NUOVO - REND_BASE
            )

            DELTA_RISK_EFF = (
                RISK_NUOVO - RISK_BASE
            )

            classification, _ = classify_allocation(
                DELTA_REND_EFF,
                DELTA_RISK_EFF
            )

            results.append({
                "AssetAumentato": asset_increased,
                "AssetRidotto": asset_reduced,
                "DeltaRendimento": DELTA_REND_EFF,
                "DeltaRischio": DELTA_RISK_EFF,
                "RendimentoNuovo": REND_NUOVO,
                "RischioNuovo": RISK_NUOVO,
                "Classificazione": classification
            })

    reallocation_table = pd.DataFrame(results)

    return (
        DELTA_REND_MARG,
        DELTA_RISK_MARG,
        OPPORTUNITA_MARG,
        TRADEOFF_MARG,
        reallocation_table
    )


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("Portfolio Optimization Dashboard")

st.caption(
    "Efficient frontier | Monte Carlo | SCR | Duration Matching "
    "| Marginal SCR | Reallocation Analysis"
)

# %% SIDEBAR

with st.sidebar:

    st.header("Model settings")

    min_weight_pct = st.number_input(
        "Minimum weight (%)",
        min_value=0.0,
        max_value=100.0,
        value=MIN_WEIGHT * 100,
        step=1.0
    )

    max_weight_pct = st.number_input(
        "Maximum weight (%)",
        min_value=0.1,
        max_value=100.0,
        value=MAX_WEIGHT * 100,
        step=1.0
    )

    st.divider()

    st.header("Liability assumptions")

    duration_liab = st.number_input(
        "Liability duration",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

    asset_pct = st.number_input(
        "Assets (%)",
        min_value=0.01,
        max_value=100.0,
        value=100.0,
        step=1.0
    )

    liability_pct = st.number_input(
        "Liabilities (%)",
        min_value=0.0,
        max_value=200.0,
        value=100.0,
        step=1.0
    )

    st.divider()

    st.header("Reallocation settings")

    shift_pct = st.number_input(
        "Reallocation amount (%)",
        min_value=0.1,
        max_value=20.0,
        value=1.0,
        step=0.5
    ) / 100.0

# %% MARKET INPUTS

st.subheader("1. Market assumptions")

st.write(
    "Expected Return and Volatility are expressed in percentage. "
    "Duration is expressed in years."
)

if "assets_input" not in st.session_state:
    st.session_state.assets_input = DEFAULT_ASSETS.copy()

edited_assets = st.data_editor(
    st.session_state.assets_input,
    num_rows="dynamic",
    use_container_width=True,
    key="assets_editor",
    column_config={
        "Asset": st.column_config.TextColumn(
            "Asset name",
            required=True
        ),
        "Expected Return (%)": st.column_config.NumberColumn(
            "Expected Return (%)",
            format="%.3f"
        ),
        "Volatility (%)": st.column_config.NumberColumn(
            "SCR Market (%)",
            format="%.3f"
        ),
        "Duration": st.column_config.NumberColumn(
            "Duration",
            format="%.3f"
        )
    }
)

st.session_state.assets_input = edited_assets.copy()

if edited_assets.empty or len(edited_assets) < 2:
    st.warning("Insert at least two asset classes.")
    st.stop()

asset_names = edited_assets["Asset"].astype(str).tolist()

if (
    len(set(asset_names)) != len(asset_names)
    or any(not x.strip() for x in asset_names)
):
    st.error("Asset names must be unique and non-empty.")
    st.stop()

numeric_cols = [
    "Expected Return (%)",
    "Volatility (%)",
    "Duration"
]

try:
    numeric_data = edited_assets[numeric_cols].astype(float)
except Exception:
    st.error("Market assumptions must contain numeric values.")
    st.stop()

if (
    numeric_data.isna().any().any()
    or not np.isfinite(numeric_data.values).all()
    or (numeric_data["Volatility (%)"] <= 0).any()
    or (numeric_data["Duration"] < 0).any()
):
    st.error(
        "Check market assumptions: no missing values, "
        "positive volatility and non-negative duration required."
    )
    st.stop()

n_assets = len(asset_names)

# %% CORRELATION MATRIX

st.subheader("2. Correlation matrix")

st.caption(
    "Enter correlations between -1 and 1. "
    "The diagonal must equal 1 and the matrix must be symmetric."
)

default_corr = np.full(
    (n_assets, n_assets),
    0.50
)

np.fill_diagonal(default_corr, 1.0)

if (
    "corr_input" not in st.session_state
    or st.session_state.get("corr_asset_names") != asset_names
):
    st.session_state.corr_input = pd.DataFrame(
        default_corr,
        index=asset_names,
        columns=asset_names
    )

    st.session_state.corr_asset_names = asset_names.copy()

corr_edited = st.data_editor(
    st.session_state.corr_input,
    use_container_width=True,
    key="corr_editor",
    column_config={
        col: st.column_config.NumberColumn(
            col,
            min_value=-1.0,
            max_value=1.0,
            format="%.3f"
        )
        for col in asset_names
    }
)

st.session_state.corr_input = corr_edited.copy()

# %% VALIDATION

try:

    corr = corr_edited.astype(float).to_numpy()

    mu = (
        numeric_data["Expected Return (%)"].to_numpy()
        / 100.0
        / ANNUALIZATION_FACTOR
    )

    vol = (
        numeric_data["Volatility (%)"].to_numpy()
        / 100.0
    )

    durations = pd.Series(
        numeric_data["Duration"].to_numpy(),
        index=asset_names
    )

    er = pd.Series(
        mu,
        index=asset_names
    )

    vol_series = pd.Series(
        vol,
        index=asset_names
    )

    corr_df = pd.DataFrame(
        corr,
        index=asset_names,
        columns=asset_names
    )

except Exception:
    st.error("Invalid numerical inputs.")
    st.stop()

if corr.shape != (n_assets, n_assets):
    st.error("Invalid correlation matrix dimensions.")
    st.stop()

if not np.isfinite(corr).all():
    st.error("Correlation matrix contains missing or invalid values.")
    st.stop()

if np.any(corr < -1) or np.any(corr > 1):
    st.error("Correlations must be between -1 and 1.")
    st.stop()

if not np.allclose(corr, corr.T, atol=1e-6):
    st.error("Correlation matrix must be symmetric.")
    st.stop()

if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
    st.error("The diagonal must be equal to 1.")
    st.stop()

# Original model does not apply a nearest PSD adjustment.
eigenvalues = np.linalg.eigvalsh(corr)

if eigenvalues.min() < -1e-8:
    st.error(
        "The correlation matrix is not positive semidefinite. "
        "The original model does not adjust the matrix automatically. "
        "Please revise the correlation assumptions."
    )
    st.stop()

min_weight = min_weight_pct / 100.0
max_weight = max_weight_pct / 100.0

if (
    min_weight * n_assets > TARGET_SUM + 1e-9
    or max_weight * n_assets < TARGET_SUM - 1e-9
):
    st.error("Weight constraints do not allow a 100% allocation.")
    st.stop()

if asset_pct <= 0:
    st.error("Asset percentage must be positive.")
    st.stop()

# Original covariance matrix
cov = pd.DataFrame(
    corr_to_cov(corr_df, vol_series.values)
    / ANNUALIZATION_FACTOR,
    index=asset_names,
    columns=asset_names
)

duration_target = (
    duration_liab * liability_pct / asset_pct
)

with st.expander("Input diagnostics"):

    st.write("Correlation matrix used")

    st.dataframe(
        corr_df.style.format("{:.3f}"),
        use_container_width=True
    )

    st.write("Covariance matrix used")

    st.dataframe(
        cov.style.format("{:.6f}"),
        use_container_width=True
    )

    st.write(
        f"**Target duration:** {duration_target:.3f}"
    )

    st.write(
        f"**Minimum correlation eigenvalue:** "
        f"{eigenvalues.min():.8f}"
    )

# %% MODEL EXECUTION

st.divider()

run_col, info_col = st.columns([1, 3])

with run_col:

    run_model = st.button(
        "Run optimization",
        type="primary",
        use_container_width=True
    )

with info_col:

    st.caption(
        "The model runs only when you press Run optimization. "
        "The original methodology is preserved."
    )

if run_model:

    if min_weight_pct != 0.0 or max_weight_pct != 100.0:
        st.warning(
            "Weight constraints differ from the original model settings."
        )

    progress_bar = st.progress(
        0,
        text="Running Monte Carlo simulations..."
    )

    try:

        def update_progress(current, total):

            progress_bar.progress(
                current / total,
                text=f"Completed simulation {current}/{total}"
            )

        # Original Monte Carlo simulation
        weights_simulations, risk_simulations, return_simulations = simulate(
            er=er,
            cov=cov,
            n_simulations=int(n_simulations),
            n_observations=SIMULATION,
            durations=durations,
            duration_liab=float(duration_liab),
            liability_perc=float(liability_pct),
            asset_perc=float(asset_pct),
            n_portfolios=int(n_portfolios),
            gamma_min=float(gamma_min),
            gamma_max=float(gamma_max),
            lambda_duration=float(lambda_duration),
            seed=int(seed) if use_seed else None,
            progress_callback=update_progress
        )

        # Original aggregation
        (
            avg_weights,
            avg_risk,
            avg_ret,
            exp_returns,
            duration_asset,
            duration_mismatch,
            duration_target,
            opt
        ) = aggregate_simulations(
            weights_simulations,
            risk_simulations,
            return_simulations,
            er,
            durations,
            duration_liab,
            liability_pct,
            asset_pct
        )

        # Original marginal SCR and Return / mSCR
        avg_mSCR, avg_return_per_mscr = calculate_marginal_averages(
            weights_simulations,
            risk_simulations,
            vol_series,
            corr_df,
            er
        )

        result = {
            "avg_weights": avg_weights,
            "avg_risk": avg_risk,
            "avg_ret": avg_ret,
            "exp_returns": exp_returns,
            "duration_asset": duration_asset,
            "duration_mismatch": duration_mismatch,
            "duration_target": duration_target,
            "opt": opt,
            "avg_mSCR": avg_mSCR,
            "avg_return_per_mscr": avg_return_per_mscr,
            "weights_simulations": weights_simulations,
            "risk_simulations": risk_simulations,
            "return_simulations": return_simulations,
            "asset_names": asset_names.copy(),
            "er": er.copy(),
            "vol": vol_series.copy(),
            "corr": corr_df.copy(),
            "cov": cov.copy(),
            "durations": durations.copy(),
            "n_simulations": int(n_simulations),
            "n_portfolios": int(n_portfolios),
            "seed": int(seed) if use_seed else None,
            "inputs": {
                "duration_liab": float(duration_liab),
                "liability_pct": float(liability_pct),
                "asset_pct": float(asset_pct),
                "gamma_min": float(gamma_min),
                "gamma_max": float(gamma_max),
                "lambda_duration": float(lambda_duration),
                "min_weight": min_weight,
                "max_weight": max_weight
            }
        }

        st.session_state.optimization_result = result

        progress_bar.empty()

        st.success("Optimization completed successfully.")

    except Exception as exc:

        progress_bar.empty()

        st.error(
            f"Optimization error: {exc}"
        )

        st.stop()

# ============================================================
# OUTPUT
# ============================================================

if "optimization_result" not in st.session_state:

    st.info(
        "Insert or modify your inputs and press Run optimization "
        "to generate the efficient frontier."
    )

    st.stop()

res = st.session_state.optimization_result

# Avoid displaying stale results after input changes
inputs_changed = (
    res["asset_names"] != asset_names
    or not np.allclose(res["er"].values, er.values)
    or not np.allclose(res["vol"].values, vol_series.values)
    or not np.allclose(res["corr"].values, corr_df.values)
    or not np.allclose(res["durations"].values, durations.values)
    or res["inputs"]["duration_liab"] != float(duration_liab)
    or res["inputs"]["liability_pct"] != float(liability_pct)
    or res["inputs"]["asset_pct"] != float(asset_pct)
    or res["inputs"]["min_weight"] != min_weight
    or res["inputs"]["max_weight"] != max_weight
)

if inputs_changed:

    st.warning(
        "Inputs have changed since the last optimization. "
        "Press Run optimization to update the results."
    )

    st.stop()

# %% RESULTS

avg_weights = res["avg_weights"]
avg_risk = res["avg_risk"]
avg_ret = res["avg_ret"]
exp_returns = res["exp_returns"]

duration_asset = res["duration_asset"]
duration_mismatch = res["duration_mismatch"]

avg_mSCR = res["avg_mSCR"]
avg_return_per_mscr = res["avg_return_per_mscr"]

asset_names = res["asset_names"]

n_port = len(avg_risk)

# %% FRONTIER DATAFRAME

frontier_df = pd.DataFrame({
    "Portfolio": np.arange(1, n_port + 1),
    "SCR (%)": avg_risk.values * 100,
    "Expected Return (%)": exp_returns.values * 100,
    "Simulated Return (%)": avg_ret.values * 100,
    "Asset Duration": duration_asset.values,
    "Liability Duration": res["inputs"]["duration_liab"],
    "Duration Mismatch": duration_mismatch.values
})

# %% PORTFOLIO SELECTION

st.divider()

st.header("3. Efficient frontier")

left, right = st.columns([1, 2])

with left:

    selected_portfolio = st.number_input(
        "Select portfolio",
        min_value=1,
        max_value=n_port,
        value=n_port,
        step=1
    )

idx = int(selected_portfolio) - 1

selected_weights = avg_weights.iloc[idx].values

selected_risk = float(avg_risk.iloc[idx])

selected_return = float(exp_returns.iloc[idx])

selected_sim_return = float(avg_ret.iloc[idx])

selected_duration = float(duration_asset.iloc[idx])

selected_mismatch = float(duration_mismatch.iloc[idx])

with right:

    st.caption(
        f"Portfolio {selected_portfolio} of {n_port} | "
        f"Average across {res['n_simulations']} simulations"
    )

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Expected Return",
    f"{selected_return:.2%}"
)

k2.metric(
    "Portfolio SCR",
    f"{selected_risk:.2%}"
)

k3.metric(
    "Asset Duration",
    f"{selected_duration:.2f}"
)

k4.metric(
    "Duration Mismatch",
    f"{selected_mismatch:.2f}"
)

# %% TABS

(
    tab_data,
    tab_overview,
    tab_portfolio,
    tab_marginal,
    tab_reallocation
) = st.tabs([
    "Portfolio Weights",
    "Graph",
    "Portfolio analysis",
    "Marginal SCR",
    "Reallocation analysis"
])

# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab_overview:

    st.subheader("Portfolio allocation and efficient frontier")

    # Original colors
    colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, len(asset_names))
    )

    # --------------------------------------------------------
    # 1. PORTFOLIO ALLOCATION ACROSS RISK LEVELS
    # --------------------------------------------------------

    st.markdown("**Portfolio Allocation across Risk Levels**")

    weights_by_risk = avg_weights.copy()

    weights_by_risk.index = avg_risk.values

    weights_by_risk = weights_by_risk.sort_index()

    fig, ax = plt.subplots(figsize=(13, 6))

    weights_by_risk.clip(0, 1).plot.area(
        ax=ax,
        stacked=True,
        color=colors,
        linewidth=0
    )

    ax.set_ylim(0, 1)

    ax.set_title(
        "Portfolio Allocation across Risk Levels",
        pad=15,
        weight="bold"
    )

    ax.set_xlabel("Portfolio SCR")

    ax.set_ylabel("Portfolio Weights")

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )

    ax.xaxis.set_major_locator(
        plt.MaxNLocator(10)
    )

    ax.xaxis.set_major_formatter(
        mtick.FormatStrFormatter("%.2f")
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False
    )

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # 2. ASSET ALLOCATION BY PORTFOLIO RISK
    # --------------------------------------------------------

    st.markdown("**Asset Allocation by Portfolio Risk**")

    weights_by_risk_rounded = avg_weights.copy()

    weights_by_risk_rounded.index = (
        avg_risk.round(2).values
    )

    weights_by_risk_rounded = (
        weights_by_risk_rounded
        .groupby(level=0)
        .mean()
    )

    weights_by_risk_rounded = (
        weights_by_risk_rounded.sort_index()
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    weights_by_risk_rounded.plot(
        ax=ax,
        color=colors,
        linewidth=2
    )

    ax.set_title(
        "Asset Allocation by Portfolio Risk",
        pad=15,
        weight="bold"
    )

    ax.set_xlabel("Portfolio SCR")

    ax.set_ylabel("Portfolio Weights")

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )

    ax.xaxis.set_major_locator(
        plt.MaxNLocator(10)
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False
    )

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # 3. EFFICIENT FRONTIER
    # --------------------------------------------------------

    st.markdown("**Efficient Frontier – SCR vs Expected Return**")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        avg_risk.values,
        exp_returns.values,
        linewidth=2.5,
        marker="o",
        markersize=4
    )

    ax.scatter(
        [selected_risk],
        [selected_return],
        marker="*",
        s=180,
        zorder=5,
        label=f"Selected portfolio {selected_portfolio}"
    )

    ax.set_title(
        "Efficient Frontier – SCR vs Expected Return",
        pad=15,
        weight="bold"
    )

    ax.set_xlabel("Portfolio SCR")

    ax.set_ylabel("Expected Return")

    ax.xaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )

    ax.xaxis.set_major_locator(
        plt.MaxNLocator(10)
    )

    ax.yaxis.set_major_locator(
        plt.MaxNLocator(8)
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # SELECTED PORTFOLIO ALLOCATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        f"Portfolio {selected_portfolio} – Asset Allocation"
    )

    selected_allocation = pd.DataFrame({
        "Asset": asset_names,
        "Weight (%)": selected_weights * 100
    }).sort_values(
        "Weight (%)",
        ascending=False
    )

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.4 * len(asset_names)))
    )

    ax.barh(
        selected_allocation["Asset"][::-1],
        selected_allocation["Weight (%)"][::-1]
    )

    ax.set_xlabel("Weight (%)")

    ax.set_title(
        f"Portfolio {selected_portfolio} – Asset Allocation",
        weight="bold"
    )

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# ============================================================
# TAB 2 - PORTFOLIO ANALYSIS
# ============================================================

with tab_portfolio:

    st.subheader(
        f"Portfolio {selected_portfolio} – Composition"
    )

    allocation_table = pd.DataFrame({
        "Asset": asset_names,
        "Weight": selected_weights,
        "Expected Return": er.values,
        "Volatility": vol_series.values,
        "Duration": durations.values,
        "Contribution to Return": selected_weights * er.values
    })

    st.dataframe(
        allocation_table.style.format({
            "Weight": "{:.2%}",
            "Expected Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Duration": "{:.2f}",
            "Contribution to Return": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Portfolio metrics")

    metrics = pd.DataFrame({
        "Metric": [
            "Expected Return",
            "Simulated Expected Return",
            "Portfolio SCR",
            "Asset Duration",
            "Liability Duration Target",
            "Duration Mismatch"
        ],
        "Value": [
            selected_return,
            selected_sim_return,
            selected_risk,
            selected_duration,
            res["duration_target"],
            selected_mismatch
        ]
    })

    st.dataframe(
        metrics.style.format({
            "Value": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 - MARGINAL SCR
# ============================================================

with tab_marginal:

    st.subheader(
        "Marginal SCR and Return per Marginal SCR"
    )

    # Original Monte Carlo averages
    selected_mSCR = avg_mSCR.iloc[idx]

    selected_return_mscr = avg_return_per_mscr.iloc[idx]

    marginal_df = pd.DataFrame({
        "Asset": asset_names,
        "Weight (%)": selected_weights * 100,
        "Marginal SCR": selected_mSCR.values,
        "Return / mSCR": selected_return_mscr.values
    }).sort_values(
        "Marginal SCR",
        ascending=False
    )

    st.dataframe(
        marginal_df.style.format({
            "Weight (%)": "{:.2f}%",
            "Marginal SCR": "{:.6f}",
            "Return / mSCR": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    c1, c2 = st.columns(2)

    with c1:

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            marginal_df["Asset"],
            marginal_df["Marginal SCR"]
        )

        ax.set_title(
            "Marginal SCR by Asset",
            weight="bold"
        )

        ax.tick_params(
            axis="x",
            rotation=75
        )

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    with c2:

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            marginal_df["Asset"],
            marginal_df["Return / mSCR"]
        )

        ax.set_title(
            "Expected Return per Marginal SCR",
            weight="bold"
        )

        ax.tick_params(
            axis="x",
            rotation=75
        )

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    st.divider()

    st.subheader("Marginal SCR – All Portfolios")

    all_marginal = avg_mSCR.copy()

    all_marginal.insert(
        0,
        "Portfolio",
        np.arange(1, n_port + 1)
    )

    st.dataframe(
        all_marginal.style.format({
            a: "{:.6f}" for a in asset_names
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Return / mSCR – All Portfolios")

    all_return_mscr = avg_return_per_mscr.copy()

    all_return_mscr.insert(
        0,
        "Portfolio",
        np.arange(1, n_port + 1)
    )

    st.dataframe(
        all_return_mscr.style.format({
            a: "{:.4f}" for a in asset_names
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 4 - REALLOCATION ANALYSIS
# ============================================================

with tab_reallocation:

    st.subheader("One-for-one Reallocation Analysis")

    st.caption(
        "Rows represent the asset increased; columns represent "
        "the asset reduced."
    )

    # Original portfolio data
    W0 = avg_weights.iloc[idx].values

    EXP_RET = (
        er.loc[asset_names].values
        * ANNUALIZATION_FACTOR
    )

    COV = (
        cov.loc[asset_names, asset_names].values
        * ANNUALIZATION_FACTOR
    )

    RISK_BASE = np.sqrt(
        W0 @ COV @ W0
    )

    # Original selected portfolio marginal SCR
    mSCR_selected = calculate_marginal_scr(
        avg_weights.iloc[[idx]],
        vol_series.loc[asset_names],
        corr_df.loc[asset_names, asset_names],
        np.array([RISK_BASE])
    ).iloc[0].values

    (
        DELTA_REND_MARG,
        DELTA_RISK_MARG,
        OPPORTUNITA_MARG,
        TRADEOFF_MARG,
        reallocation_table
    ) = calculate_reallocation_analysis(
        W0=W0,
        asset_names=asset_names,
        EXP_RET=EXP_RET,
        COV=COV,
        mSCR=mSCR_selected,
        shift=shift_pct,
        max_weight=max_weight,
        min_weight=min_weight
    )

    # --------------------------------------------------------
    # MARGINAL RETURN DIFFERENCE
    # --------------------------------------------------------

    st.markdown("**1. Marginal Return Difference**")

    st.caption(
        "Rows: asset increased. Columns: asset reduced."
    )

    st.dataframe(
        DELTA_REND_MARG.style.format("{:+.4%}"),
        use_container_width=True
    )

    # --------------------------------------------------------
    # MARGINAL RISK DIFFERENCE
    # --------------------------------------------------------

    st.markdown("**2. Marginal SCR Difference**")

    st.dataframe(
        DELTA_RISK_MARG.style.format("{:+.6f}"),
        use_container_width=True
    )

    # --------------------------------------------------------
    # OPPORTUNITY MATRIX
    # --------------------------------------------------------

    st.markdown("**3. Reallocation Opportunity Matrix**")

    st.dataframe(
        OPPORTUNITA_MARG,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TRADE-OFF MATRIX
    # --------------------------------------------------------

    st.markdown("**4. Trade-off Matrix**")

    st.dataframe(
        TRADEOFF_MARG,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ACTUAL REALLOCATION TABLE
    # --------------------------------------------------------

    st.divider()

    st.markdown("**5. Effective Reallocation Analysis**")

    st.caption(
        f"Reallocation shift: {shift_pct:.2%} | "
        f"Selected portfolio: {selected_portfolio} | "
        f"Maximum weight: {max_weight:.0%}"
    )

    display_reallocation = reallocation_table.copy()

    display_reallocation["DeltaRendimento (pp)"] = (
        display_reallocation["DeltaRendimento"] * 100
    )

    display_reallocation["DeltaRischio (pp)"] = (
        display_reallocation["DeltaRischio"] * 100
    )

    display_reallocation["RendimentoNuovo (%)"] = (
        display_reallocation["RendimentoNuovo"] * 100
    )

    display_reallocation["RischioNuovo (%)"] = (
        display_reallocation["RischioNuovo"] * 100
    )

    display_reallocation = display_reallocation[[
        "AssetAumentato",
        "AssetRidotto",
        "DeltaRendimento (pp)",
        "DeltaRischio (pp)",
        "RendimentoNuovo (%)",
        "RischioNuovo (%)",
        "Classificazione"
    ]]

    st.dataframe(
        display_reallocation.style.format({
            "DeltaRendimento (pp)": "{:+.4f}",
            "DeltaRischio (pp)": "{:+.4f}",
            "RendimentoNuovo (%)": "{:.4f}",
            "RischioNuovo (%)": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5 - DATA TABLES
# ============================================================

with tab_data:

    st.subheader("Frontier – All Portfolios")

    st.dataframe(
        frontier_df.style.format({
            "SCR (%)": "{:.4f}",
            "Expected Return (%)": "{:.4f}",
            "Simulated Return (%)": "{:.4f}",
            "Asset Duration": "{:.4f}",
            "Liability Duration": "{:.4f}",
            "Duration Mismatch": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Weights – All Portfolios")

    weights_table = avg_weights.copy() * 100

    weights_table.insert(
        0,
        "Portfolio",
        np.arange(1, n_port + 1)
    )

    st.dataframe(
        weights_table.style.format({
            a: "{:.2f}%" for a in asset_names
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Complete Optimization Results")

    st.dataframe(
        res["opt"].style.format("{:.6f}"),
        use_container_width=True
    )

    st.divider()

    st.subheader("Download results")

    csv_frontier = frontier_df.to_csv(
        index=False
    ).encode("utf-8")

    csv_weights = weights_table.to_csv(
        index=False
    ).encode("utf-8")

    csv_marginal = avg_mSCR.to_csv().encode("utf-8")

    csv_return_mscr = (
        avg_return_per_mscr.to_csv().encode("utf-8")
    )

    csv_reallocation = (
        reallocation_table.to_csv(index=False).encode("utf-8")
    )

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        "Download frontier CSV",
        csv_frontier,
        "frontier_results.csv",
        "text/csv",
        use_container_width=True
    )

    d2.download_button(
        "Download weights CSV",
        csv_weights,
        "portfolio_weights.csv",
        "text/csv",
        use_container_width=True
    )

    d3.download_button(
        "Download marginal SCR CSV",
        csv_marginal,
        "marginal_scr.csv",
        "text/csv",
        use_container_width=True
    )

    d4, d5 = st.columns(2)

    d4.download_button(
        "Download Return / mSCR CSV",
        csv_return_mscr,
        "return_per_mscr.csv",
        "text/csv",
        use_container_width=True
    )

    d5.download_button(
        "Download reallocation CSV",
        csv_reallocation,
        "reallocation_analysis.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# METHODOLOGICAL NOTE
# ============================================================

st.divider()

st.caption(
    "Methodology: The efficient frontier is estimated through "
    "Monte Carlo simulations of multivariate normally distributed "
    "asset returns. For each simulation, the model generates "
    "50 observations and optimizes 100 portfolios by maximizing "
    "Expected Return − Gamma × Variance − Lambda × Duration "
    "Mismatch². Portfolio weights, SCR and simulated returns are "
    "averaged across simulations. Expected Return is calculated "
    "using the average portfolio weights and original expected "
    "return assumptions. Marginal SCR and Return / mSCR follow "
    "the original Monte Carlo aggregation procedure. "
    "Reallocation analysis evaluates one-for-one asset shifts "
    "using the original covariance and marginal risk framework."
)
