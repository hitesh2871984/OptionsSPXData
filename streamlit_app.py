import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetStatus,
    ContractType,
    OrderClass,
    OrderSide,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    OptionLegRequest,
)

load_dotenv()

st.set_page_config(
    page_title="Alpaca ^GSPC / SPX All Options Strategies",
    layout="wide",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"], .stMarkdown, p, label, span, div, button, input {
            font-size: 14px !important;
        }
        h1, h2, h3, h4, h5, h6, .stSubheader, .stTitle {
            font-size: 14px !important;
            line-height: 1.35 !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricLabel"] {
            font-size: 14px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🦅 Alpaca ^GSPC (SPX) — One-Leg & Multi-Leg Options Suite")

STRATEGY_CATALOG = [
    # One-Leg
    "1-Leg: Long Call (Debit)",
    "1-Leg: Short Call (Credit)",
    "1-Leg: Long Put (Debit)",
    "1-Leg: Short Put (Credit)",
    # Iron Condors
    "Iron Condor: Short (Credit)",
    "Iron Condor: Long (Debit)",
    "Iron Condor: Both (Credit + Debit)",
    # Butterflies
    "Butterfly: Long Call (Debit)",
    "Butterfly: Short Call (Credit)",
    "Butterfly: Long Put (Debit)",
    "Butterfly: Short Put (Credit)",
    "Butterfly: Iron Butterfly (Credit)",
    "Butterfly: Inverse Iron Butterfly (Debit)",
    "Butterfly: Both Credit + Debit Call Butterflies",
    "Butterfly: Both Credit + Debit Put Butterflies",
    # Vertical Spreads & Strangles
    "Vertical: Bull Call Spread (Debit)",
    "Vertical: Bear Call Spread (Credit)",
    "Vertical: Bull Put Spread (Credit)",
    "Vertical: Bear Put Spread (Debit)",
    "Strangle: Long Strangle (Debit)",
    "Strangle: Short Strangle (Credit)",
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Alpaca API Credentials")
    api_key = st.text_input(
        "API Key", value=os.getenv("ALPACA_API_KEY", ""), type="password"
    )
    secret_key = st.text_input(
        "Secret Key", value=os.getenv("ALPACA_SECRET_KEY", ""), type="password"
    )
    paper_mode = st.checkbox("Paper Trading Mode", value=True)

    st.divider()
    st.header("2. Strategy Selection")
    selected_strategies = st.multiselect(
        "Select One or More Strategies (Individual or Combined)",
        options=STRATEGY_CATALOG,
        default=["Iron Condor: Short (Credit)", "Iron Condor: Long (Debit)"],
    )

    st.divider()
    st.header("3. Underlying & Expiration")
    index_ticker = st.text_input("Spot Index Ticker", value="^GSPC").strip()
    option_underlier = st.text_input("Alpaca Option Underlier", value="SPX").upper().strip()
    manual_spot = st.number_input(
        "Manual Spot Override (0 = Auto ^GSPC)", min_value=0.0, value=0.0, step=5.0
    )
    dte_range = st.slider("Target DTE Window (Days)", 0, 45, (1, 7))
    default_qty = st.number_input("Default Contracts (Qty)", min_value=1, value=1, step=1)

    st.divider()
    st.header("4. Strike Offsets & Widths (pts)")
    one_leg_dist = st.number_input(
        "1-Leg Strike Offset from Spot", min_value=0.0, value=25.0, step=5.0
    )
    vertical_inner_dist = st.number_input(
        "Vertical Short/Long Inner Dist", min_value=0.0, value=25.0, step=5.0
    )
    vertical_width = st.number_input(
        "Vertical Spread Width", min_value=5.0, value=15.0, step=5.0
    )
    strangle_dist = st.number_input(
        "Strangle Call/Put OTM Offset", min_value=5.0, value=60.0, step=5.0
    )

    st.subheader("Iron Condor Params")
    ic_credit_dist = st.number_input(
        "IC Credit Inner Dist", min_value=5.0, value=85.0, step=5.0
    )
    ic_credit_width = st.number_input(
        "IC Credit Wing Width", min_value=5.0, value=10.0, step=5.0
    )
    ic_debit_dist = st.number_input(
        "IC Debit Inner Dist", min_value=5.0, value=35.0, step=5.0
    )
    ic_debit_width = st.number_input(
        "IC Debit Wing Width", min_value=5.0, value=10.0, step=5.0
    )

    st.subheader("Butterfly Params")
    bf_center_offset = st.number_input(
        "Butterfly Center Offset from Spot", min_value=-200.0, value=0.0, step=5.0
    )
    bf_wing_width = st.number_input(
        "Butterfly Wing Width (Center to Outer)", min_value=5.0, value=25.0, step=5.0
    )
    bf_credit_center_offset = st.number_input(
        "Credit Butterfly Center Offset", min_value=-200.0, value=0.0, step=5.0
    )
    bf_credit_width = st.number_input(
        "Credit Butterfly Wing Width", min_value=5.0, value=25.0, step=5.0
    )
    bf_debit_center_offset = st.number_input(
        "Debit Butterfly Center Offset", min_value=-200.0, value=50.0, step=5.0
    )
    bf_debit_width = st.number_input(
        "Debit Butterfly Wing Width", min_value=5.0, value=25.0, step=5.0
    )


def get_clients(key: str, secret: str, paper: bool):
    return (
        TradingClient(api_key=key, secret_key=secret, paper=paper),
        OptionHistoricalDataClient(api_key=key, secret_key=secret),
    )


def get_gspc_spot(ticker: str, override: float) -> float:
    if override > 0:
        return float(override)
    data = yf.Ticker(ticker).history(period="1d", interval="1m")
    if data.empty:
        data = yf.Ticker(ticker).history(period="5d")
    if data.empty:
        raise ValueError(f"Could not fetch {ticker}. Provide a Manual Spot Override.")
    return float(data["Close"].iloc[-1])


def nearest(contracts, target_strike: float):
    return min(contracts, key=lambda c: abs(float(c.strike_price) - target_strike))


def leg_payoff_at_expiry(
    prices: np.ndarray, ctype: ContractType, strike: float, side: OrderSide, ratio: int
) -> np.ndarray:
    if ctype == ContractType.CALL:
        intrinsic = np.maximum(prices - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - prices, 0.0)
    sign = 1.0 if side == OrderSide.BUY else -1.0
    return sign * ratio * intrinsic


def make_sleeve(name: str, legs_spec: list, quotes: dict, qty: int):
    """Build a strategy sleeve with mid cashflow, strikes, and leg table rows."""
    rows = []
    net_per_share = 0.0
    strikes = []

    for contract, side, ratio in legs_spec:
        q = quotes[contract.symbol]
        bid, ask = float(q.bid_price), float(q.ask_price)
        mid = round((bid + ask) / 2.0, 2)
        strike = float(contract.strike_price)
        strikes.append(strike)

        signed = mid * ratio if side == OrderSide.SELL else -mid * ratio
        net_per_share += signed

        rows.append(
            {
                "Strategy": name,
                "Action": side.value.upper(),
                "Ratio": ratio,
                "Qty": qty * ratio,
                "Contract": contract.symbol,
                "Type": contract.type.value.upper(),
                "Strike": strike,
                "Bid": bid,
                "Ask": ask,
                "Mid": mid,
            }
        )

    return {
        "name": name,
        "qty": qty,
        "legs": legs_spec,
        "rows": rows,
        "strikes": strikes,
        "net_per_share": round(net_per_share, 2),
        "net_dollars": round(net_per_share * 100 * qty, 2),
    }


def sleeve_pnl_curve(prices: np.ndarray, sleeve: dict) -> np.ndarray:
    total = np.full_like(prices, sleeve["net_per_share"], dtype=float)
    for contract, side, ratio in sleeve["legs"]:
        total += leg_payoff_at_expiry(
            prices, contract.type, float(contract.strike_price), side, ratio
        )
    return total * 100.0 * sleeve["qty"]


def find_breakeven_strikes(prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Find underlying strike levels where expiration P&L crosses zero."""
    breakevens = []
    for i in range(len(prices) - 1):
        y1, y2 = pnl[i], pnl[i + 1]
        x1, x2 = prices[i], prices[i + 1]
        if y1 == 0.0:
            breakevens.append(round(float(x1), 2))
        elif (y1 < 0 < y2) or (y1 > 0 > y2):
            x_zero = x1 + (-y1) * (x2 - x1) / (y2 - y1)
            breakevens.append(round(float(x_zero), 2))

    deduped = []
    for be in sorted(breakevens):
        if not deduped or abs(be - deduped[-1]) > 0.25:
            deduped.append(be)
    return deduped


def build_sleeves_for_strategy(
    strat: str, puts: list, calls: list, quotes: dict, spot: float, qty: int
) -> list:
    sleeves = []

    # --- 1-LEG STRATEGIES ---
    if strat == "1-Leg: Long Call (Debit)":
        c = nearest(calls, spot + one_leg_dist)
        sleeves.append(make_sleeve(strat, [(c, OrderSide.BUY, 1)], quotes, qty))
    elif strat == "1-Leg: Short Call (Credit)":
        c = nearest(calls, spot + one_leg_dist)
        sleeves.append(make_sleeve(strat, [(c, OrderSide.SELL, 1)], quotes, qty))
    elif strat == "1-Leg: Long Put (Debit)":
        p = nearest(puts, spot - one_leg_dist)
        sleeves.append(make_sleeve(strat, [(p, OrderSide.BUY, 1)], quotes, qty))
    elif strat == "1-Leg: Short Put (Credit)":
        p = nearest(puts, spot - one_leg_dist)
        sleeves.append(make_sleeve(strat, [(p, OrderSide.SELL, 1)], quotes, qty))

    # --- IRON CONDORS ---
    elif strat in (
        "Iron Condor: Short (Credit)",
        "Iron Condor: Long (Debit)",
        "Iron Condor: Both (Credit + Debit)",
    ):
        def ic_legs(dist, width, is_short):
            ip = nearest(puts, spot - dist)
            op = nearest(puts, float(ip.strike_price) - width)
            ic = nearest(calls, spot + dist)
            oc = nearest(calls, float(ic.strike_price) + width)
            if is_short:
                return [
                    (op, OrderSide.BUY, 1),
                    (ip, OrderSide.SELL, 1),
                    (ic, OrderSide.SELL, 1),
                    (oc, OrderSide.BUY, 1),
                ]
            return [
                (op, OrderSide.SELL, 1),
                (ip, OrderSide.BUY, 1),
                (ic, OrderSide.BUY, 1),
                (oc, OrderSide.SELL, 1),
            ]

        if strat in ("Iron Condor: Short (Credit)", "Iron Condor: Both (Credit + Debit)"):
            sleeves.append(
                make_sleeve(
                    "Short IC (Credit)",
                    ic_legs(ic_credit_dist, ic_credit_width, True),
                    quotes,
                    qty,
                )
            )
        if strat in ("Iron Condor: Long (Debit)", "Iron Condor: Both (Credit + Debit)"):
            sleeves.append(
                make_sleeve(
                    "Long IC (Debit)",
                    ic_legs(ic_debit_dist, ic_debit_width, False),
                    quotes,
                    qty,
                )
            )

    # --- CALL BUTTERFLIES ---
    elif strat in (
        "Butterfly: Long Call (Debit)",
        "Butterfly: Short Call (Credit)",
        "Butterfly: Both Credit + Debit Call Butterflies",
    ):
        def call_bf_legs(center_off, width, is_long):
            mid_k = float(nearest(calls, spot + center_off).strike_price)
            low_c = nearest(calls, mid_k - width)
            mid_c = nearest(calls, mid_k)
            high_c = nearest(calls, mid_k + width)
            if is_long:
                return [
                    (low_c, OrderSide.BUY, 1),
                    (mid_c, OrderSide.SELL, 2),
                    (high_c, OrderSide.BUY, 1),
                ]
            return [
                (low_c, OrderSide.SELL, 1),
                (mid_c, OrderSide.BUY, 2),
                (high_c, OrderSide.SELL, 1),
            ]

        if strat in (
            "Butterfly: Short Call (Credit)",
            "Butterfly: Both Credit + Debit Call Butterflies",
        ):
            off = bf_credit_center_offset if "Both" in strat else bf_center_offset
            w = bf_credit_width if "Both" in strat else bf_wing_width
            sleeves.append(
                make_sleeve(
                    "Short Call Butterfly (Credit)",
                    call_bf_legs(off, w, False),
                    quotes,
                    qty,
                )
            )
        if strat in (
            "Butterfly: Long Call (Debit)",
            "Butterfly: Both Credit + Debit Call Butterflies",
        ):
            off = bf_debit_center_offset if "Both" in strat else bf_center_offset
            w = bf_debit_width if "Both" in strat else bf_wing_width
            sleeves.append(
                make_sleeve(
                    "Long Call Butterfly (Debit)",
                    call_bf_legs(off, w, True),
                    quotes,
                    qty,
                )
            )

    # --- PUT BUTTERFLIES ---
    elif strat in (
        "Butterfly: Long Put (Debit)",
        "Butterfly: Short Put (Credit)",
        "Butterfly: Both Credit + Debit Put Butterflies",
    ):
        def put_bf_legs(center_off, width, is_long):
            mid_k = float(nearest(puts, spot + center_off).strike_price)
            low_p = nearest(puts, mid_k - width)
            mid_p = nearest(puts, mid_k)
            high_p = nearest(puts, mid_k + width)
            if is_long:
                return [
                    (low_p, OrderSide.BUY, 1),
                    (mid_p, OrderSide.SELL, 2),
                    (high_p, OrderSide.BUY, 1),
                ]
            return [
                (low_p, OrderSide.SELL, 1),
                (mid_p, OrderSide.BUY, 2),
                (high_p, OrderSide.SELL, 1),
            ]

        if strat in (
            "Butterfly: Short Put (Credit)",
            "Butterfly: Both Credit + Debit Put Butterflies",
        ):
            off = bf_credit_center_offset if "Both" in strat else bf_center_offset
            w = bf_credit_width if "Both" in strat else bf_wing_width
            sleeves.append(
                make_sleeve(
                    "Short Put Butterfly (Credit)",
                    put_bf_legs(off, w, False),
                    quotes,
                    qty,
                )
            )
        if strat in (
            "Butterfly: Long Put (Debit)",
            "Butterfly: Both Credit + Debit Put Butterflies",
        ):
            off = bf_debit_center_offset if "Both" in strat else bf_center_offset
            w = bf_debit_width if "Both" in strat else bf_wing_width
            sleeves.append(
                make_sleeve(
                    "Long Put Butterfly (Debit)",
                    put_bf_legs(off, w, True),
                    quotes,
                    qty,
                )
            )

    # --- IRON BUTTERFLIES ---
    elif strat in (
        "Butterfly: Iron Butterfly (Credit)",
        "Butterfly: Inverse Iron Butterfly (Debit)",
    ):
        center_k = float(nearest(calls, spot + bf_center_offset).strike_price)
        sc = nearest(calls, center_k)
        sp = nearest(puts, center_k)
        oc = nearest(calls, center_k + bf_wing_width)
        op = nearest(puts, center_k - bf_wing_width)
        if "Credit" in strat:
            legs = [
                (op, OrderSide.BUY, 1),
                (sp, OrderSide.SELL, 1),
                (sc, OrderSide.SELL, 1),
                (oc, OrderSide.BUY, 1),
            ]
        else:
            legs = [
                (op, OrderSide.SELL, 1),
                (sp, OrderSide.BUY, 1),
                (sc, OrderSide.BUY, 1),
                (oc, OrderSide.SELL, 1),
            ]
        sleeves.append(make_sleeve(strat, legs, quotes, qty))

    # --- VERTICAL SPREADS ---
    elif strat == "Vertical: Bull Call Spread (Debit)":
        lc = nearest(calls, spot + vertical_inner_dist)
        sc = nearest(calls, float(lc.strike_price) + vertical_width)
        sleeves.append(
            make_sleeve(
                strat, [(lc, OrderSide.BUY, 1), (sc, OrderSide.SELL, 1)], quotes, qty
            )
        )
    elif strat == "Vertical: Bear Call Spread (Credit)":
        sc = nearest(calls, spot + vertical_inner_dist)
        lc = nearest(calls, float(sc.strike_price) + vertical_width)
        sleeves.append(
            make_sleeve(
                strat, [(sc, OrderSide.SELL, 1), (lc, OrderSide.BUY, 1)], quotes, qty
            )
        )
    elif strat == "Vertical: Bull Put Spread (Credit)":
        sp = nearest(puts, spot - vertical_inner_dist)
        lp = nearest(puts, float(sp.strike_price) - vertical_width)
        sleeves.append(
            make_sleeve(
                strat, [(sp, OrderSide.SELL, 1), (lp, OrderSide.BUY, 1)], quotes, qty
            )
        )
    elif strat == "Vertical: Bear Put Spread (Debit)":
        lp = nearest(puts, spot - vertical_inner_dist)
        sp = nearest(puts, float(lp.strike_price) - vertical_width)
        sleeves.append(
            make_sleeve(
                strat, [(lp, OrderSide.BUY, 1), (sp, OrderSide.SELL, 1)], quotes, qty
            )
        )

    # --- STRANGLES ---
    elif strat == "Strangle: Long Strangle (Debit)":
        p = nearest(puts, spot - strangle_dist)
        c = nearest(calls, spot + strangle_dist)
        sleeves.append(
            make_sleeve(strat, [(p, OrderSide.BUY, 1), (c, OrderSide.BUY, 1)], quotes, qty)
        )
    elif strat == "Strangle: Short Strangle (Credit)":
        p = nearest(puts, spot - strangle_dist)
        c = nearest(calls, spot + strangle_dist)
        sleeves.append(
            make_sleeve(
                strat, [(p, OrderSide.SELL, 1), (c, OrderSide.SELL, 1)], quotes, qty
            )
        )

    return sleeves


# --- MAIN WORKSPACE ---
if not api_key or not secret_key:
    st.caption(
        "Tip: Add your Alpaca API Key and Secret Key in the sidebar or `.env` file before scanning."
    )

if not selected_strategies:
    st.caption("Tip: Select at least one strategy in the sidebar.")

if st.button(
    "🔍 Scan ^GSPC / SPX Chain & Build Selected Strategies",
    type="primary",
    key="btn_scan_chain",
):
    if not api_key or not secret_key:
        st.error("Please enter your Alpaca API Key and Secret Key in the sidebar first.")
    elif not selected_strategies:
        st.error("Please select at least one strategy in the sidebar first.")
    else:
        try:
            t_client, o_client = get_clients(api_key, secret_key, paper_mode)
            spot = get_gspc_spot(index_ticker, manual_spot)

            max_offset = (
                max(
                    one_leg_dist,
                    vertical_inner_dist + vertical_width,
                    strangle_dist,
                    ic_credit_dist + ic_credit_width,
                    ic_debit_dist + ic_debit_width,
                    abs(bf_center_offset) + bf_wing_width,
                    abs(bf_credit_center_offset) + bf_credit_width,
                    abs(bf_debit_center_offset) + bf_debit_width,
                )
                + 50.0
            )

            ny_today = datetime.now(ZoneInfo("America/New_York")).date()
            req = GetOptionContractsRequest(
                underlying_symbols=[option_underlier],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=ny_today + timedelta(days=dte_range[0]),
                expiration_date_lte=ny_today + timedelta(days=dte_range[1]),
                strike_price_gte=str(max(100.0, spot - max_offset)),
                strike_price_lte=str(spot + max_offset),
            )
            contracts = t_client.get_option_contracts(req).option_contracts
            if not contracts:
                raise ValueError(
                    "No active option contracts found for the selected DTE/strike window."
                )

            target_exp = sorted({c.expiration_date for c in contracts})[0]
            exp_contracts = [c for c in contracts if c.expiration_date == target_exp]
            spxw = [c for c in exp_contracts if c.symbol.startswith("SPXW")]
            if spxw:
                exp_contracts = spxw

            puts = [c for c in exp_contracts if c.type == ContractType.PUT]
            calls = [c for c in exp_contracts if c.type == ContractType.CALL]

            quotes = o_client.get_option_latest_quote(
                OptionLatestQuoteRequest(
                    symbol_or_symbols=[c.symbol for c in exp_contracts]
                )
            )

            all_sleeves = []
            for strat in selected_strategies:
                all_sleeves.extend(
                    build_sleeves_for_strategy(
                        strat, puts, calls, quotes, spot, default_qty
                    )
                )

            st.session_state["portfolio_state"] = {
                "spot": spot,
                "expiration": target_exp,
                "sleeves": all_sleeves,
                "strategies": selected_strategies,
            }
        except Exception as e:
            st.error(f"Error building strategies: {e}")

if "portfolio_state" in st.session_state:
    pstate = st.session_state["portfolio_state"]
    spot = pstate["spot"]
    sleeves = pstate["sleeves"]

    strikes_all = [k for s in sleeves for k in s["strikes"]]
    low_x = min(strikes_all + [spot]) - 60.0
    high_x = max(strikes_all + [spot]) + 60.0
    prices = np.linspace(low_x, high_x, 2000)

    combined_pnl = np.zeros_like(prices)
    for s in sleeves:
        combined_pnl += sleeve_pnl_curve(prices, s)

    breakevens = find_breakeven_strikes(prices, combined_pnl)
    be_text = (
        ", ".join(f"{be:,.2f}" for be in breakevens) if breakevens else "None in range"
    )
    net_cashflow_dollars = round(sum(s["net_dollars"] for s in sleeves), 2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{index_ticker} Spot", f"{spot:,.2f}")
    m2.metric("Expiration", str(pstate["expiration"]))
    cf_title = "Total Net Credit" if net_cashflow_dollars >= 0 else "Total Net Debit"
    m3.metric(cf_title, f"${abs(net_cashflow_dollars):,.2f}")
    m4.metric(
        "Profit / Loss & Breakevens",
        f"+${combined_pnl.max():,.0f} / ${combined_pnl.min():,.0f}",
        delta=f"BE Strikes: {be_text}",
        delta_color="off",
    )

    left_col, right_col = st.columns([0.4, 0.6])

    with left_col:
        st.subheader("Legs & Order Execution")
        df_all = pd.DataFrame([r for s in sleeves for r in s["rows"]])
        st.dataframe(df_all, width="stretch", hide_index=True)

        limit_overrides = {}
        for idx, s in enumerate(sleeves):
            default_lim = round(-s["net_per_share"], 2)
            limit_overrides[s["name"]] = st.number_input(
                f"{s['name']} Limit Price (Neg=Credit, Pos=Debit)",
                value=default_lim,
                step=0.05,
                format="%.2f",
                key=f"lim_all_{idx}",
            )

        if st.button(
            "🚀 Submit Selected Strategy Orders to Alpaca",
            type="primary",
            width="stretch",
            key="btn_submit_orders",
        ):
            try:
                t_client, _ = get_clients(api_key, secret_key, paper_mode)
                for s in sleeves:
                    lim = limit_overrides[s["name"]]
                    if len(s["legs"]) == 1:
                        contract, side, ratio = s["legs"][0]
                        req = LimitOrderRequest(
                            symbol=contract.symbol,
                            qty=s["qty"] * ratio,
                            side=side,
                            time_in_force=TimeInForce.DAY,
                            limit_price=abs(lim),
                        )
                    else:
                        legs_req = [
                            OptionLegRequest(
                                symbol=c.symbol, side=side, ratio_qty=ratio
                            )
                            for c, side, ratio in s["legs"]
                        ]
                        req = LimitOrderRequest(
                            qty=s["qty"],
                            order_class=OrderClass.MLEG,
                            time_in_force=TimeInForce.DAY,
                            limit_price=lim,
                            legs=legs_req,
                        )
                    resp = t_client.submit_order(order_data=req)
                    st.success(f"{s['name']} Submitted! ID: `{resp.id}`")
            except Exception as e:
                st.error(f"Order Error: {e}")

    with right_col:
        fig = go.Figure()

        if len(sleeves) > 1:
            for s in sleeves:
                curve = sleeve_pnl_curve(prices, s)
                s_be = find_breakeven_strikes(prices, curve)
                s_be_str = ", ".join(f"{b:,.2f}" for b in s_be) if s_be else "None"
                fig.add_trace(
                    go.Scatter(
                        x=prices,
                        y=curve,
                        mode="lines",
                        name=f"{s['name']} (${s['net_dollars']:+,.0f} | BE: {s_be_str})",
                        line=dict(dash="dash", width=2),
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=prices,
                y=combined_pnl,
                mode="lines",
                name=f"Total Combined P&L (+${combined_pnl.max():,.0f} / ${combined_pnl.min():,.0f} | BE: {be_text})",
                line=dict(color="#1f77b4", width=3.2),
            )
        )

        if breakevens:
            fig.add_trace(
                go.Scatter(
                    x=breakevens,
                    y=[0.0] * len(breakevens),
                    mode="markers+text",
                    name=f"Breakeven Strikes ({be_text})",
                    text=[f"BE: {be:,.2f}" for be in breakevens],
                    textposition="bottom center",
                    textfont=dict(size=12),
                    hovertemplate="Breakeven Strike: %{x:,.2f}<br>P&L: $0.00<extra></extra>",
                    marker=dict(color="#d62728", size=9, symbol="diamond"),
                )
            )
            for be in breakevens:
                fig.add_vline(
                    x=be,
                    line_dash="dashdot",
                    line_color="#d62728",
                    line_width=1.2,
                )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(
            x=spot,
            line_dash="dot",
            line_color="black",
            annotation_text=f"Spot: {spot:,.2f}",
            annotation_position="bottom right",
            annotation_font=dict(size=12),
        )
        fig.update_layout(
            font=dict(size=12),
            xaxis=dict(
                title=dict(text="^GSPC / SPX Settlement Level", font=dict(size=13)),
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title=dict(text="Profit / Loss ($)", font=dict(size=13)),
                tickfont=dict(size=12),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
            height=520,
            margin=dict(l=20, r=20, t=20, b=110),
        )
        st.plotly_chart(fig, width="stretch")
