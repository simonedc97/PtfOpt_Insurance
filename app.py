import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import cvxpy as cp
import warnings

# ============================================================
# PORTFOLIO OPTIMIZATION DASHBOARD
# Manual inputs only - no Excel dependency
# ============================================================

st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📊",
    layout="wide"
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

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
    "Expected Return (%)": [4.9, 6.0, 5.5, 7.1, 4.8, 3.89, 4.5, 3.7, 5.2, 10.0],
    "Volatility (%)": [16.0, 30.2, 18.495, 26.167, 21.0, 11.0, 46.0, 22.78, 22.0, 22.0],
    "Duration": [4.0, 4.5, 4.5, 5.0, 5.0, 3.0, 2.0, 3.0, 2.5, 7.0],
})

def nearest_psd(matrix, epsilon=1e-8):
    """Project a symmetric matrix to positive semidefinite."""
    matrix = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, epsilon)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return (out + out.T) / 2

def portfolio_risk(weights, covariance):
    value = float(weights @ covariance @ weights)
    return np.sqrt(max(value, 0.0))

def optimize_frontier(mu, covariance, durations, duration_target,
                      n_portfolios, gamma_min, gamma_max,
                      lambda_duration, min_weight, max_weight):
    n_assets = len(mu)
    w = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)

    risk_variance = cp.quad_form(w, cp.psd_wrap(covariance))
    expected_return = mu @ w
    duration_mismatch = durations @ w - duration_target

    objective = cp.Maximize(
        expected_return
        - gamma * risk_variance
        - lambda_duration * cp.square(duration_mismatch)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]
    problem = cp.Problem(objective, constraints)

    gamma_values = np.logspace(gamma_min, gamma_max, n_portfolios)
    all_weights, all_returns, all_risks = [], [], []

    for gamma_value in gamma_values:
        gamma.value = float(gamma_value)
        solved = False
        for solver in [cp.CLARABEL, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
                    solved = True
                    break
            except Exception:
                pass
        if not solved:
            raise RuntimeError(
                f"Ottimizzazione non convergente per gamma={gamma_value:.6g}. "
                "Verificare input, correlazioni e vincoli."
            )

        weights = np.asarray(w.value).flatten()
        all_weights.append(weights)
        all_returns.append(float(mu @ weights))
        all_risks.append(portfolio_risk(weights, covariance))

    return (
        np.asarray(all_weights),
        np.asarray(all_returns),
        np.asarray(all_risks),
        gamma_values,
    )

def calculate_marginal_scr(weights, vol, corr, risk):
    if risk <= 1e-12:
        return np.zeros_like(vol)
    return vol * (corr @ (weights * vol)) / risk

def classify_allocation(delta_return, delta_risk, tol=1e-12):
    if delta_return > tol and delta_risk < -tol:
        return "Aumento rendimento / diminuzione rischio"
    if delta_return > tol and delta_risk > tol:
        return "Aumento rendimento / aumento rischio"
    if delta_return < -tol and delta_risk < -tol:
        return "Diminuzione rendimento / diminuzione rischio"
    if delta_return < -tol and delta_risk > tol:
        return "Diminuzione rendimento / aumento rischio"
    return "Effetto nullo o invariato"

# --------------------------- SIDEBAR ---------------------------
st.title("Portfolio Optimization Dashboard")
st.caption("Efficient frontier, SCR, duration matching and portfolio reallocation")

with st.sidebar:
    st.header("Model settings")
    n_simulations = st.number_input("Monte Carlo simulations", 1, 500, 50, 1)
    n_portfolios = st.number_input("Portfolios on the frontier", 10, 300, 100, 5)
    gamma_min = st.number_input("Gamma minimum (log10)", value=-3.0, step=0.5)
    gamma_max = st.number_input("Gamma maximum (log10)", value=3.0, step=0.5)
    lambda_duration = st.number_input("Lambda duration", min_value=0.0, value=1.0, step=0.1)
    min_weight_pct = st.number_input("Minimum weight (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    max_weight_pct = st.number_input("Maximum weight (%)", min_value=0.1, max_value=100.0, value=20.0, step=1.0)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    st.divider()
    st.header("Liability assumptions")
    duration_liab = st.number_input("Liability duration", min_value=0.0, value=7.0, step=0.1)
    asset_pct = st.number_input("Assets (%)", min_value=0.01, max_value=100.0, value=100.0, step=1.0)
    liability_pct = st.number_input("Liabilities (%)", min_value=0.0, max_value=200.0, value=100.0, step=1.0)

# --------------------------- INPUT DATA ---------------------------
st.subheader("1. Market assumptions")
st.write("Modifica direttamente la tabella. Expected Return e Volatility sono espresse in percentuale.")

if "assets_input" not in st.session_state:
    st.session_state.assets_input = DEFAULT_ASSETS.copy()

edited_assets = st.data_editor(
    st.session_state.assets_input,
    num_rows="dynamic",
    use_container_width=True,
    key="assets_editor",
    column_config={
        "Asset": st.column_config.TextColumn("Asset name", required=True),
        "Expected Return (%)": st.column_config.NumberColumn("Expected Return (%)", format="%.3f"),
        "Volatility (%)": st.column_config.NumberColumn("Volatility (%)", format="%.3f"),
        "Duration": st.column_config.NumberColumn("Duration", format="%.3f"),
    },
)
st.session_state.assets_input = edited_assets.copy()

if edited_assets.empty or len(edited_assets) < 2:
    st.warning("Inserire almeno due asset class.")
    st.stop()

asset_names = edited_assets["Asset"].astype(str).tolist()
if len(set(asset_names)) != len(asset_names) or any(not x.strip() for x in asset_names):
    st.error("I nomi degli asset devono essere valorizzati e univoci.")
    st.stop()

numeric_cols = ["Expected Return (%)", "Volatility (%)", "Duration"]
try:
    numeric_data = edited_assets[numeric_cols].astype(float)
except Exception:
    st.error("Expected Return, Volatility e Duration devono essere numerici.")
    st.stop()

if numeric_data.isna().any().any() or (numeric_data["Volatility (%)"] <= 0).any():
    st.error("Verificare i dati: nessun valore mancante e volatilità strettamente positiva.")
    st.stop()

n_assets = len(asset_names)
default_corr = np.full((n_assets, n_assets), 0.50)
np.fill_diagonal(default_corr, 1.0)

if ("corr_input" not in st.session_state
        or st.session_state.get("corr_asset_names") != asset_names):
    st.session_state.corr_input = pd.DataFrame(default_corr, index=asset_names, columns=asset_names)
    st.session_state.corr_asset_names = asset_names.copy()

st.subheader("2. Correlation matrix")
st.caption("Inserire correlazioni comprese tra -1 e 1. La diagonale deve essere pari a 1 e la matrice simmetrica.")
corr_edited = st.data_editor(
    st.session_state.corr_input,
    use_container_width=True,
    key="corr_editor",
    column_config={
        col: st.column_config.NumberColumn(col, min_value=-1.0, max_value=1.0, format="%.3f")
        for col in asset_names
    },
)
st.session_state.corr_input = corr_edited.copy()

# --------------------------- VALIDATION ---------------------------
try:
    corr = corr_edited.astype(float).to_numpy()
    mu = numeric_data["Expected Return (%)"].to_numpy() / 100.0
    vol = numeric_data["Volatility (%)"].to_numpy() / 100.0
    durations = numeric_data["Duration"].to_numpy()
except Exception:
    st.error("La matrice di correlazione e gli input devono contenere solo valori numerici.")
    st.stop()

if corr.shape != (n_assets, n_assets) or not np.isfinite(corr).all():
    st.error("La matrice di correlazione contiene valori mancanti o dimensioni non coerenti.")
    st.stop()
if np.any(corr < -1) or np.any(corr > 1):
    st.error("Le correlazioni devono essere comprese tra -1 e 1.")
    st.stop()
if not np.allclose(corr, corr.T, atol=1e-6):
    st.error("La matrice di correlazione deve essere simmetrica.")
    st.stop()
if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
    st.error("La diagonale della matrice di correlazione deve essere pari a 1.")
    st.stop()
if gamma_max <= gamma_min:
    st.error("Gamma maximum deve essere maggiore di Gamma minimum.")
    st.stop()

min_weight = min_weight_pct / 100.0
max_weight = max_weight_pct / 100.0
if min_weight * n_assets > 1 + 1e-9 or max_weight * n_assets < 1 - 1e-9:
    st.error("I limiti minimi/massimi di peso non consentono di investire il 100% del portafoglio.")
    st.stop()
if asset_pct <= 0:
    st.error("La percentuale degli asset deve essere maggiore di zero.")
    st.stop()

corr_psd = nearest_psd(corr)
was_adjusted = not np.allclose(corr, corr_psd, atol=1e-5)
covariance = np.outer(vol, vol) * corr_psd
duration_target = duration_liab * (liability_pct / asset_pct)

if was_adjusted:
    st.warning(
        "La matrice inserita non è semidefinita positiva. Per consentire il calcolo "
        "è stata applicata una correzione numerica alla matrice. La matrice utilizzata "
        "è mostrata nella sezione Input diagnostics."
    )

with st.expander("Input diagnostics – correlation matrix used"):
    st.dataframe(pd.DataFrame(corr_psd, index=asset_names, columns=asset_names).style.format("{:.3f}"),
                 use_container_width=True)
    st.write(f"Target duration: **{duration_target:.3f}**")

# --------------------------- RUN MODEL ---------------------------
st.divider()
run_col, info_col = st.columns([1, 3])
with run_col:
    run_model = st.button("Run optimization", type="primary", use_container_width=True)
with info_col:
    st.caption("Il calcolo viene eseguito quando premi il pulsante. Modificare gli input non avvia automaticamente una nuova simulazione.")

if run_model:
    np.random.seed(int(seed))
    progress = st.progress(0, text="Running Monte Carlo simulations...")
    simulation_weights = []
    simulation_returns = []
    simulation_risks = []

    try:
        for sim in range(int(n_simulations)):
            simulated_returns = np.random.multivariate_normal(
                mean=mu,
                cov=covariance,
                size=50,
                check_valid="raise",
            )
            simulated_mu = simulated_returns.mean(axis=0)
            simulated_cov = np.cov(simulated_returns, rowvar=False, ddof=1)
            simulated_cov = nearest_psd(simulated_cov)

            weights, returns, risks, gammas = optimize_frontier(
                simulated_mu,
                simulated_cov,
                durations,
                duration_target,
                int(n_portfolios),
                float(gamma_min),
                float(gamma_max),
                float(lambda_duration),
                min_weight,
                max_weight,
            )
            simulation_weights.append(weights)
            simulation_returns.append(returns)
            simulation_risks.append(risks)
            progress.progress((sim + 1) / int(n_simulations),
                              text=f"Completed simulation {sim + 1}/{int(n_simulations)}")

        weights_array = np.mean(np.stack(simulation_weights, axis=0), axis=0)
        risk_array = np.mean(np.stack(simulation_risks, axis=0), axis=0)
        simulated_return_array = np.mean(np.stack(simulation_returns, axis=0), axis=0)
        # Expected return is measured using the original input assumptions.
        expected_return_array = weights_array @ mu
        duration_asset_array = weights_array @ durations
        mismatch_array = duration_asset_array - duration_target

        result = {
            "weights": weights_array,
            "risk": risk_array,
            "sim_return": simulated_return_array,
            "expected_return": expected_return_array,
            "duration_asset": duration_asset_array,
            "duration_mismatch": mismatch_array,
            "asset_names": asset_names,
            "mu": mu,
            "vol": vol,
            "corr": corr_psd,
            "covariance": covariance,
            "duration_target": duration_target,
            "n_simulations": int(n_simulations),
            "n_portfolios": int(n_portfolios),
            "seed": int(seed),
        }
        st.session_state.optimization_result = result
        progress.empty()
        st.success("Ottimizzazione completata.")
    except Exception as exc:
        progress.empty()
        st.error(f"Errore durante l'ottimizzazione: {exc}")

# --------------------------- OUTPUT ---------------------------
if "optimization_result" not in st.session_state:
    st.info("Inserisci o modifica gli input e premi **Run optimization** per generare la frontiera.")
    st.stop()

res = st.session_state.optimization_result
# If input dimensions/names changed after last run, avoid showing stale results as current.
if res["asset_names"] != asset_names:
    st.warning("Gli asset sono cambiati dopo l'ultimo calcolo. Premi Run optimization per aggiornare i risultati.")
    st.stop()

weights = res["weights"]
risk = res["risk"]
expected_return = res["expected_return"]
duration_asset = res["duration_asset"]
duration_mismatch = res["duration_mismatch"]
n_port = len(risk)

st.divider()
st.header("3. Efficient frontier")

frontier_df = pd.DataFrame({
    "Portfolio": np.arange(1, n_port + 1),
    "SCR (%)": risk * 100,
    "Expected Return (%)": expected_return * 100,
    "Simulated Return (%)": res["sim_return"] * 100,
    "Asset Duration": duration_asset,
    "Duration Mismatch": duration_mismatch,
})
left, right = st.columns([1, 2])
with left:
    selected_portfolio = st.number_input(
        "Select portfolio",
        min_value=1,
        max_value=n_port,
        value=min(47, n_port),
        step=1,
        help="Seleziona il numero del portafoglio sulla frontiera, ad esempio 47."
    )
idx = int(selected_portfolio) - 1
selected_weights = weights[idx]
selected_risk = risk[idx]
selected_return = expected_return[idx]
selected_duration = duration_asset[idx]
selected_mismatch = duration_mismatch[idx]

with right:
    st.caption(f"Portfolio {selected_portfolio} of {n_port} – average weights across {res['n_simulations']} simulations")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Expected Return", f"{selected_return:.2%}")
k2.metric("Portfolio SCR", f"{selected_risk:.2%}")
k3.metric("Asset Duration", f"{selected_duration:.2f}")
k4.metric("Duration Mismatch", f"{selected_mismatch:.2f}")

tab_overview, tab_portfolio, tab_marginal, tab_reallocation, tab_data = st.tabs([
    "Overview", "Portfolio analysis", "Marginal SCR", "Reallocation analysis", "Data tables"
])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(risk * 100, expected_return * 100, marker="o", markersize=3, linewidth=1.5)
        ax.scatter([selected_risk * 100], [selected_return * 100], s=100, marker="*", zorder=5,
                   label=f"Selected portfolio {selected_portfolio}")
        ax.set_title("Efficient Frontier – SCR vs Expected Return", weight="bold")
        ax.set_xlabel("Portfolio SCR (%)")
        ax.set_ylabel("Expected Return (%)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        allocation_df = pd.DataFrame(weights, columns=asset_names)
        allocation_df["SCR"] = risk
        allocation_df = allocation_df.sort_values("SCR")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stackplot(allocation_df["SCR"] * 100,
                     [allocation_df[a].clip(0, 1).to_numpy() for a in asset_names],
                     labels=asset_names, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title("Portfolio Allocation across Risk Levels", weight="bold")
        ax.set_xlabel("Portfolio SCR (%)")
        ax.set_ylabel("Portfolio Weights")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Selected portfolio allocation")
    selected_allocation = pd.DataFrame({
        "Asset": asset_names,
        "Weight (%)": selected_weights * 100,
    }).sort_values("Weight (%)", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * n_assets)))
    ax.barh(selected_allocation["Asset"][::-1], selected_allocation["Weight (%)"][::-1])
    ax.set_xlabel("Weight (%)")
    ax.set_title(f"Portfolio {selected_portfolio} – Asset Allocation", weight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_portfolio:
    st.subheader(f"Portfolio {selected_portfolio} – composition")
    allocation_table = pd.DataFrame({
        "Asset": asset_names,
        "Weight": selected_weights,
        "Expected Return": mu,
        "Volatility": vol,
        "Duration": durations,
        "Contribution to Return": selected_weights * mu,
    })
    st.dataframe(
        allocation_table.style.format({
            "Weight": "{:.2%}", "Expected Return": "{:.2%}",
            "Volatility": "{:.2%}", "Duration": "{:.2f}",
            "Contribution to Return": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Portfolio metrics")
    metrics = pd.DataFrame({
        "Metric": ["Expected Return", "Portfolio SCR", "Asset Duration",
                   "Liability Duration Target", "Duration Mismatch"],
        "Value": [selected_return, selected_risk, selected_duration,
                  duration_target, selected_mismatch],
    })
    st.dataframe(metrics, use_container_width=True, hide_index=True)

with tab_marginal:
    st.subheader("Marginal SCR and Return per marginal SCR")
    m_scr = calculate_marginal_scr(selected_weights, vol, corr_psd, selected_risk)
    return_per_mscr = np.divide(mu, m_scr, out=np.full_like(mu, np.nan), where=np.abs(m_scr) > 1e-12)
    marginal_df = pd.DataFrame({
        "Asset": asset_names,
        "Weight (%)": selected_weights * 100,
        "Marginal SCR": m_scr,
        "Return / mSCR": return_per_mscr,
    }).sort_values("Marginal SCR", ascending=False)
    st.dataframe(
        marginal_df.style.format({
            "Weight (%)": "{:.2f}%", "Marginal SCR": "{:.4f}",
            "Return / mSCR": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(marginal_df["Asset"], marginal_df["Marginal SCR"])
        ax.set_title("Marginal SCR by Asset", weight="bold")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(marginal_df["Asset"], marginal_df["Return / mSCR"])
        ax.set_title("Expected Return per Marginal SCR", weight="bold")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab_reallocation:
    st.subheader("One-for-one reallocation analysis")
    shift_pct = st.number_input("Reallocation amount (%)", min_value=0.1, max_value=20.0,
                                value=1.0, step=0.5) / 100.0
    rows = []
    base_return = selected_weights @ mu
    base_risk = portfolio_risk(selected_weights, covariance)
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                continue
            new_weights = selected_weights.copy()
            new_weights[i] += shift_pct
            new_weights[j] -= shift_pct
            feasible = (
                new_weights[i] <= max_weight + 1e-10
                and new_weights[j] >= min_weight - 1e-10
            )
            if not feasible:
                rows.append({
                    "Asset increased": asset_names[i],
                    "Asset reduced": asset_names[j],
                    "Delta Return (pp)": np.nan,
                    "Delta SCR (pp)": np.nan,
                    "Classification": "Not feasible",
                })
                continue
            new_return = new_weights @ mu
            new_risk = portfolio_risk(new_weights, covariance)
            delta_return = new_return - base_return
            delta_risk = new_risk - base_risk
            rows.append({
                "Asset increased": asset_names[i],
                "Asset reduced": asset_names[j],
                "Delta Return (pp)": delta_return * 100,
                "Delta SCR (pp)": delta_risk * 100,
                "Classification": classify_allocation(delta_return, delta_risk),
            })
    reallocation_df = pd.DataFrame(rows)
    st.dataframe(
        reallocation_df.style.format({
            "Delta Return (pp)": "{:+.3f}",
            "Delta SCR (pp)": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab_data:
    st.subheader("Frontier – all portfolios")
    st.dataframe(frontier_df.style.format({
        "SCR (%)": "{:.3f}",
        "Expected Return (%)": "{:.3f}",
        "Simulated Return (%)": "{:.3f}",
        "Asset Duration": "{:.3f}",
        "Duration Mismatch": "{:.3f}",
    }), use_container_width=True, hide_index=True)

    st.subheader("Weights – all portfolios")
    weights_table = pd.DataFrame(weights * 100, columns=asset_names)
    weights_table.insert(0, "Portfolio", np.arange(1, n_port + 1))
    st.dataframe(weights_table.style.format({a: "{:.2f}%" for a in asset_names}),
                 use_container_width=True, hide_index=True)

    csv_frontier = frontier_df.to_csv(index=False).encode("utf-8")
    csv_weights = weights_table.to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button("Download frontier CSV", csv_frontier, "frontier_results.csv", "text/csv")
    d2.download_button("Download weights CSV", csv_weights, "portfolio_weights.csv", "text/csv")

st.caption(
    "Methodological note: the dashboard estimates the efficient frontier by repeatedly "
    "sampling annual asset returns from a multivariate normal distribution. For each "
    "simulation it optimizes the objective Return − Gamma × Variance − Lambda × "
    "Duration Mismatch². The displayed portfolio weights and risk/return statistics "
    "are averages across simulations. This is a model implementation and should be "
    "validated before use in production."
)
