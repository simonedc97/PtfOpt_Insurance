
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings

from dataclasses import dataclass
from io import BytesIO


# ============================================================
# CONFIGURAZIONE
# ============================================================

DEFAULT_FRONTIER_POINTS = 100
SIMULATION = 50
ANNUALIZATION_FACTOR = 1
GAMMA_MIN = -3
GAMMA_MAX = 3
LAMBDA_DURATION = 1
SIM = 100


# ============================================================
# CONFIGURAZIONE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide"
)

st.title("Portfolio Optimization Dashboard")
st.caption(
    "Streamlit implementation of the original Python model"
)


# ============================================================
# CLASSI ORIGINALI
# ============================================================

@dataclass
class PortfolioConstraints:
    long_only: bool = True
    max_weight: float = 1.0
    min_weight: float = 0.0
    target_sum: float = 1.0


@dataclass
class FrontierResult:
    weights: pd.DataFrame
    returns: np.ndarray
    risks: np.ndarray


class EfficientFrontier:

    def __init__(
        self,
        returns,
        duration,
        duration_liab,
        liability_perc,
        asset_perc,
        lambda_duration=1.0
    ):

        self.returns = returns
        self.mu = returns.mean() * ANNUALIZATION_FACTOR
        self.cov = returns.cov() * ANNUALIZATION_FACTOR

        self.duration = duration.loc[returns.columns].values

        self.duration_target = (
            duration_liab * liability_perc / asset_perc
        )

        self.lambda_duration = lambda_duration

        n_assets = len(returns.columns)

        self.w = cp.Variable(n_assets)
        self.gamma = cp.Parameter(nonneg=True)

        self.risk = cp.quad_form(
            self.w,
            self.cov.values
        )

        self.ret = self.mu.values.T @ self.w

        self.duration_asset = self.duration @ self.w

        self.duration_mismatch = (
            self.duration_asset - self.duration_target
        )

        objective = cp.Maximize(
            self.ret
            - self.gamma * self.risk
            - self.lambda_duration
            * cp.square(self.duration_mismatch)
        )

        self.constraints = [
            cp.sum(self.w) == PortfolioConstraints.target_sum,
            self.w >= PortfolioConstraints.min_weight,
            self.w <= PortfolioConstraints.max_weight
        ]

        self.problem = cp.Problem(
            objective,
            self.constraints
        )

    def solve(self, gamma):

        self.gamma.value = gamma

        self.problem.solve()

        return (
            np.asarray(self.w.value).flatten(),
            float(self.ret.value),
            float(np.sqrt(self.risk.value))
        )

    def generate(self):

        gamma_vals = np.logspace(
            GAMMA_MIN,
            GAMMA_MAX,
            num=DEFAULT_FRONTIER_POINTS
        )

        weights = []
        portfolio_returns = []
        portfolio_risks = []

        for gamma in gamma_vals:

            w, ret, risk = self.solve(gamma)

            weights.append(w)
            portfolio_returns.append(ret)
            portfolio_risks.append(risk)

        return FrontierResult(
            weights=pd.DataFrame(
                weights,
                columns=self.returns.columns
            ),
            returns=np.array(portfolio_returns),
            risks=np.array(portfolio_risks)
        )


# ============================================================
# FUNZIONI DI SUPPORTO ORIGINALI
# ============================================================

def simulate(
    er,
    cov,
    sims,
    duration,
    duration_liab,
    liability_perc,
    asset_perc,
    lambda_duration=1.0
):

    dates = pd.date_range(
        start='2026-09-30',
        periods=SIMULATION,
        freq='Y'
    )

    # IMPORTANTE:
    # Generazione preventiva di tutte le simulazioni,
    # esattamente come nel codice originale.
    simulations = [
        pd.DataFrame(
            np.random.multivariate_normal(
                mean=er.values,
                cov=cov.values,
                size=len(dates)
            ),
            columns=cov.columns,
            index=dates
        )
        for _ in range(sims)
    ]

    weights = []
    stdev = []
    exp_ret = []

    for simulation in simulations:

        frontier = EfficientFrontier(
            simulation,
            duration,
            duration_liab,
            liability_perc,
            asset_perc,
            lambda_duration
        )

        result = frontier.generate()

        weights.append(result.weights)
        stdev.append(result.risks)
        exp_ret.append(result.returns)

    return weights, stdev, exp_ret


def merge(*series):

    dfs = []

    for s in series:

        if isinstance(s, pd.DataFrame):
            dfs.append(s)

        elif isinstance(s, pd.Series):
            tmpdf = pd.DataFrame({s.name: s})
            dfs.append(tmpdf)

        else:
            raise NotImplementedError(
                'Unsupported merge type'
            )

    return pd.concat(dfs, axis=1)


def corr_to_cov(corr_matrix, stdevs):

    if not isinstance(corr_matrix, pd.DataFrame):

        warnings.warn(
            "cov_matrix is not a dataframe",
            RuntimeWarning
        )

        corr_matrix = pd.DataFrame(corr_matrix)

    return corr_matrix * np.outer(stdevs, stdevs)


def pltRandomMarkers(df):

    valid_markers = (
        [
            item[0]
            for item in mpl.markers.MarkerStyle.markers.items()
            if item[1] != 'nothing'
            and not item[1].startswith('tick')
            and not item[1].startswith('caret')
        ]
    )

    markers = np.random.choice(
        valid_markers,
        df.shape[1],
        replace=False
    )

    ax = df.plot(
        kind='line',
        markersize=5,
        linewidth=2
    )

    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])

    ax.legend(
        ax.get_lines(),
        df.columns,
        loc='best'
    )


def calculate_marginal_scr(
    weights,
    vol,
    corr,
    risk
):

    D = vol.values
    CORR = corr.values

    mSCR = np.zeros_like(weights.values)

    for p in range(len(weights)):

        W = weights.iloc[p].values

        SCR_COMPONENT = W * D

        mSCR[p, :] = (
            D * (CORR @ SCR_COMPONENT) / risk[p]
        )

    return pd.DataFrame(
        mSCR,
        index=weights.index,
        columns=weights.columns
    )


def calculate_return_per_marginal_scr(
    weights,
    er,
    mSCR
):

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


def classify_allocation(
    dR,
    dS,
    tol=1e-12
):

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


def calculate_reallocation_analysis(
    W0,
    asset_names,
    EXP_RET,
    COV,
    mSCR,
    SHIFT=0.01,
    max_weight=0.30,
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

    for i in range(n_asset):

        for j in range(n_asset):

            if i == j:

                OPPORTUNITA_MARG.iloc[i, j] = "N/A"
                TRADEOFF_MARG.iloc[i, j] = "N/A"

                continue

            dR = DELTA_REND[i, j]
            dS = DELTA_RISK_MARG[i, j]

            classe, tradeoff = classify_allocation(
                dR,
                dS
            )

            OPPORTUNITA_MARG.iloc[i, j] = classe
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

    REND_BASE = W0 @ EXP_RET
    RISK_BASE = np.sqrt(W0 @ COV @ W0)

    RISULTATI = []

    for i in range(n_asset):

        for j in range(n_asset):

            if i == j:
                continue

            ASSET_ACQUISTATO = asset_names[i]
            ASSET_VENDUTO = asset_names[j]

            if W0[j] - SHIFT < min_weight - 1e-12:

                RISULTATI.append({
                    'AssetAumentato': ASSET_ACQUISTATO,
                    'AssetRidotto': ASSET_VENDUTO,
                    'DeltaRendimento': np.nan,
                    'DeltaRischio': np.nan,
                    'RendimentoNuovo': np.nan,
                    'RischioNuovo': np.nan,
                    'Classificazione': 'Non fattibile'
                })

                continue

            if W0[i] + SHIFT > max_weight + 1e-12:

                RISULTATI.append({
                    'AssetAumentato': ASSET_ACQUISTATO,
                    'AssetRidotto': ASSET_VENDUTO,
                    'DeltaRendimento': np.nan,
                    'DeltaRischio': np.nan,
                    'RendimentoNuovo': np.nan,
                    'RischioNuovo': np.nan,
                    'Classificazione': 'Non fattibile'
                })

                continue

            W1 = W0.copy()

            W1[i] += SHIFT
            W1[j] -= SHIFT

            REND_NUOVO = W1 @ EXP_RET
            RISK_NUOVO = np.sqrt(W1 @ COV @ W1)

            DELTA_REND_EFF = REND_NUOVO - REND_BASE
            DELTA_RISK_EFF = RISK_NUOVO - RISK_BASE

            CLASSE, _ = classify_allocation(
                DELTA_REND_EFF,
                DELTA_RISK_EFF
            )

            RISULTATI.append({
                'AssetAumentato': ASSET_ACQUISTATO,
                'AssetRidotto': ASSET_VENDUTO,
                'DeltaRendimento': DELTA_REND_EFF,
                'DeltaRischio': DELTA_RISK_EFF,
                'RendimentoNuovo': REND_NUOVO,
                'RischioNuovo': RISK_NUOVO,
                'Classificazione': CLASSE
            })

    TABELLA_RIALLOCAZIONI = pd.DataFrame(RISULTATI)

    return (
        DELTA_REND_MARG,
        DELTA_RISK_MARG,
        OPPORTUNITA_MARG,
        TRADEOFF_MARG,
        TABELLA_RIALLOCAZIONI
    )


# ============================================================
# CARICAMENTO DATI
# ============================================================

def load_market_assumptions(excel_bytes):

    df = pd.read_excel(
        BytesIO(excel_bytes),
        sheet_name='data',
        header=0,
        index_col=0
    )

    vol = df['Vola']

    er = df['ExpRet'] / ANNUALIZATION_FACTOR

    duration = df['Dur']

    corr = pd.read_excel(
        BytesIO(excel_bytes),
        sheet_name='data',
        header=0,
        index_col=0,
        usecols='A,G:O'
    )

    corr = corr.loc[vol.index, vol.index]

    cov = (
        corr_to_cov(corr, vol)
        / ANNUALIZATION_FACTOR
    )

    df2 = pd.read_excel(
        BytesIO(excel_bytes),
        sheet_name='duration',
        header=0
    )

    duration_liab = df2['DurLiab']
    asset_perc = df2['Asset']
    liab_perc = df2['Liability']

    return (
        er,
        cov,
        vol,
        corr,
        duration,
        duration_liab,
        asset_perc,
        liab_perc
    )


# ============================================================
# CARICAMENTO FILE E AVVIO
# ============================================================

uploaded_file = st.file_uploader(
    "Carica il file data_2.xlsx",
    type=["xlsx"]
)

if uploaded_file is not None:

    st.success("File caricato correttamente.")

    if st.button(
        "Avvia simulazione",
        type="primary"
    ):

        with st.spinner(
            "Esecuzione delle simulazioni Monte Carlo e ottimizzazione..."
        ):

            # ================================================
            # INPUT
            # ================================================

            excel_bytes = uploaded_file.getvalue()

            (
                er,
                cov,
                vol,
                corr,
                duration,
                duration_liab,
                asset_perc,
                liab_perc
            ) = load_market_assumptions(excel_bytes)

            # ================================================
            # OTTIMIZZAZIONE ORIGINALE
            # ================================================

            weights, risk, exp_ret = simulate(
                er,
                cov,
                SIM,
                duration,
                duration_liab,
                liab_perc,
                asset_perc,
                LAMBDA_DURATION
            )

            avg_risk = pd.DataFrame(risk).mean()

            avg_ret = pd.DataFrame(exp_ret).mean()

            avg_weigths = (
                pd.concat(weights)
                .groupby(level=0)
                .mean()
            )

            exp_returns = (
                avg_weigths @ er
            ) * ANNUALIZATION_FACTOR

            duration_asset = (
                avg_weigths
                @ duration.loc[avg_weigths.columns]
            )

            duration_target = float(
                duration_liab * liab_perc / asset_perc
            )

            duration_mismatch = pd.Series(
                duration_asset.values - duration_target,
                index=duration_asset.index
            )

            opt = pd.DataFrame(
                merge(
                    avg_weigths,
                    avg_risk,
                    avg_ret,
                    exp_returns
                )
            )

            new_fields = [
                'Risk',
                'Exp_Ret_Sim',
                'Exp_Ret'
            ]

            opt.columns = (
                avg_weigths.columns.union(
                    new_fields,
                    sort=False
                )
            )

            opt['Duration_Asset'] = duration_asset
            opt['Duration_Liability'] = duration_liab
            opt['Duration_Mismatch'] = duration_mismatch

            # ================================================
            # STILE GRAFICI ORIGINALE
            # ================================================

            plt.rcParams.update({
                'font.family': 'Century Gothic',
                'font.size': 10,
                'axes.titlesize': 13,
                'axes.labelsize': 11,
                'legend.fontsize': 9,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.alpha': 0.20,
                'grid.linestyle': '--',
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            })

            colors = plt.get_cmap('tab20')(
                np.linspace(
                    0,
                    1,
                    len(avg_weigths.columns)
                )
            )

            # ================================================
            # GRAFICO 1
            # ================================================

            weights_by_risk = avg_weigths.copy()

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
                'Portfolio Allocation across Risk Levels',
                pad=15,
                weight='bold'
            )

            ax.set_xlabel('Portfolio SCR')
            ax.set_ylabel('Portfolio Weights')

            ax.yaxis.set_major_formatter(
                mtick.PercentFormatter(1)
            )

            ax.xaxis.set_major_locator(
                plt.MaxNLocator(10)
            )

            ax.xaxis.set_major_formatter(
                mtick.FormatStrFormatter('%.2f')
            )

            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.20),
                ncol=3,
                frameon=False
            )

            plt.tight_layout()

            # ================================================
            # GRAFICO 2
            # ================================================

            weights_by_risk_rounded = avg_weigths.copy()

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

            fig2, ax2 = plt.subplots(figsize=(13, 6))

            weights_by_risk_rounded.plot(
                ax=ax2,
                color=colors,
                linewidth=2
            )

            ax2.set_title(
                'Asset Allocation by Portfolio Risk',
                pad=15,
                weight='bold'
            )

            ax2.set_xlabel('Portfolio SCR')
            ax2.set_ylabel('Portfolio Weights')

            ax2.yaxis.set_major_formatter(
                mtick.PercentFormatter(1)
            )

            ax2.xaxis.set_major_locator(
                plt.MaxNLocator(10)
            )

            ax2.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.22),
                ncol=3,
                frameon=False
            )

            plt.tight_layout()

            # ================================================
            # GRAFICO 3
            # ================================================

            fig3, ax3 = plt.subplots(figsize=(10, 6))

            ax3.plot(
                avg_risk,
                exp_returns,
                linewidth=2.5,
                marker='o',
                markersize=4
            )

            ax3.set_title(
                'Efficient Frontier – SCR vs Expected Return',
                pad=15,
                weight='bold'
            )

            ax3.set_xlabel('Portfolio SCR')
            ax3.set_ylabel('Expected Return')

            ax3.xaxis.set_major_formatter(
                mtick.PercentFormatter(1)
            )

            ax3.yaxis.set_major_formatter(
                mtick.PercentFormatter(1)
            )

            ax3.xaxis.set_major_locator(
                plt.MaxNLocator(10)
            )

            ax3.yaxis.set_major_locator(
                plt.MaxNLocator(8)
            )

            plt.tight_layout()

            # ================================================
            # MARGINALITA' - CODICE ORIGINALE
            # ================================================

            mSCR_simulations = []
            rend_per_mscr_simulations = []

            for p in range(SIM):

                mSCR = calculate_marginal_scr(
                    weights[p],
                    vol.loc[weights[p].columns],
                    corr.loc[
                        weights[p].columns,
                        weights[p].columns
                    ],
                    risk[p]
                )

                mSCR_simulations.append(mSCR)

                mSCR = (
                    pd.concat(mSCR_simulations)
                    .groupby(level=0)
                    .mean()
                )

                rend_per_mscr = (
                    calculate_return_per_marginal_scr(
                        weights[p],
                        er,
                        mSCR
                    )
                )

                rend_per_mscr_simulations.append(
                    rend_per_mscr
                )

                rend_per_mscr = (
                    pd.concat(rend_per_mscr_simulations)
                    .groupby(level=0)
                    .mean()
                )

            # ================================================
            # TRADE-OFF ANALYSIS - CODICE ORIGINALE
            # ================================================

            portfolio_idx = 99

            ASSET_NAMES = avg_weigths.columns.tolist()

            W0 = avg_weigths.iloc[
                portfolio_idx
            ].values

            EXP_RET = (
                er.loc[ASSET_NAMES].values
                * ANNUALIZATION_FACTOR
            )

            COV = (
                cov.loc[
                    ASSET_NAMES,
                    ASSET_NAMES
                ].values
                * ANNUALIZATION_FACTOR
            )

            RISK_BASE = np.sqrt(
                W0 @ COV @ W0
            )

            mSCR_selected = calculate_marginal_scr(
                avg_weigths.iloc[[portfolio_idx]],
                vol.loc[ASSET_NAMES],
                corr.loc[
                    ASSET_NAMES,
                    ASSET_NAMES
                ],
                np.array([RISK_BASE])
            ).iloc[0].values

            (
                DELTA_REND_MARG,
                DELTA_RISK_MARG,
                OPPORTUNITA_MARG,
                tradeoff,
                riallocation
            ) = calculate_reallocation_analysis(
                W0=W0,
                asset_names=ASSET_NAMES,
                EXP_RET=EXP_RET,
                COV=COV,
                mSCR=mSCR_selected,
                SHIFT=0.01,
                max_weight=PortfolioConstraints.max_weight,
                min_weight=PortfolioConstraints.min_weight
            )

        # ====================================================
        # OUTPUT STREAMLIT
        # ====================================================

        st.success("Simulazione completata.")

        st.subheader("Model parameters")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Simulations", SIM)
        col2.metric("Observations", SIMULATION)
        col3.metric("Frontier points", DEFAULT_FRONTIER_POINTS)
        col4.metric("Lambda duration", LAMBDA_DURATION)

        st.divider()

        # --------------------------------------------
        # INPUT
        # --------------------------------------------

        with st.expander("Market assumptions", expanded=False):

            st.write("Expected Returns")
            st.dataframe(er.to_frame("ExpRet"))

            st.write("Volatility")
            st.dataframe(vol.to_frame("Vola"))

            st.write("Duration")
            st.dataframe(duration.to_frame("Duration"))

            st.write("Correlation Matrix")
            st.dataframe(corr)

            st.write("Covariance Matrix")
            st.dataframe(cov)

            st.write("Liability / Asset assumptions")

            duration_inputs = pd.DataFrame({
                "Duration Liability": duration_liab,
                "Asset": asset_perc,
                "Liability": liab_perc
            })

            st.dataframe(duration_inputs)

        # --------------------------------------------
        # OTTIMIZZAZIONE
        # --------------------------------------------

        st.header("Portfolio Optimization")

        st.subheader("Portfolio results")

        st.dataframe(
            opt.style.format("{:.4%}"),
            use_container_width=True
        )

        st.subheader("Portfolio Allocation across Risk Levels")
        st.pyplot(fig)

        st.subheader("Asset Allocation by Portfolio Risk")
        st.pyplot(fig2)

        st.subheader("Efficient Frontier – SCR vs Expected Return")
        st.pyplot(fig3)

        # --------------------------------------------
        # MARGINALITA'
        # --------------------------------------------

        st.divider()

        st.header("Marginal Risk Contribution")

        st.subheader("Marginal SCR")

        st.dataframe(
            mSCR,
            use_container_width=True
        )

        st.subheader("Return per Marginal SCR")

        st.dataframe(
            rend_per_mscr,
            use_container_width=True
        )

        # --------------------------------------------
        # TRADE-OFF
        # --------------------------------------------

        st.divider()

        st.header("Trade-off Analysis")

        st.write(
            f"Selected portfolio: {portfolio_idx + 1}"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Portfolio SCR",
            f"{RISK_BASE:.4%}"
        )

        col2.metric(
            "Expected Return",
            f"{W0 @ EXP_RET:.4%}"
        )

        st.subheader("Marginal Return Differences")
        st.dataframe(
            DELTA_REND_MARG.style.format("{:.4%}"),
            use_container_width=True
        )

        st.subheader("Marginal Risk Differences")
        st.dataframe(
            DELTA_RISK_MARG.style.format("{:.4%}"),
            use_container_width=True
        )

        st.subheader("Allocation Opportunities")
        st.dataframe(
            OPPORTUNITA_MARG,
            use_container_width=True
        )

        st.subheader("Trade-off Classification")
        st.dataframe(
            tradeoff,
            use_container_width=True
        )

        st.subheader("Reallocation Analysis")

        st.dataframe(
            riallocation.style.format({
                "DeltaRendimento": "{:.4%}",
                "DeltaRischio": "{:.4%}",
                "RendimentoNuovo": "{:.4%}",
                "RischioNuovo": "{:.4%}"
            }),
            use_container_width=True
        )

        # --------------------------------------------
        # DOWNLOAD RISULTATI
        # --------------------------------------------

        output = BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            opt.to_excel(
                writer,
                sheet_name="Portfolio Results"
            )

            avg_weigths.to_excel(
                writer,
                sheet_name="Average Weights"
            )

            pd.DataFrame(
                risk
            ).to_excel(
                writer,
                sheet_name="Risk Simulations"
            )

            pd.DataFrame(
                exp_ret
            ).to_excel(
                writer,
                sheet_name="Return Simulations"
            )

            mSCR.to_excel(
                writer,
                sheet_name="Marginal SCR"
            )

            rend_per_mscr.to_excel(
                writer,
                sheet_name="Return per mSCR"
            )

            DELTA_REND_MARG.to_excel(
                writer,
                sheet_name="Delta Return"
            )

            DELTA_RISK_MARG.to_excel(
                writer,
                sheet_name="Delta Risk"
            )

            OPPORTUNITA_MARG.to_excel(
                writer,
                sheet_name="Opportunities"
            )

            tradeoff.to_excel(
                writer,
                sheet_name="Trade-off"
            )

            riallocation.to_excel(
                writer,
                sheet_name="Reallocation",
                index=False
            )

        st.download_button(
            label="Download results Excel",
            data=output.getvalue(),
            file_name="Portfolio_Optimization_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:

    st.info(
        "Carica il file data_2.xlsx per avviare il modello."
    )
