import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from src.data.data_processor import DataProcessor
from src.optimization.portfolio_theory import PortfolioOptimizer
from src.optimization.pso_optimizer import PSOPortfolioOptimizer

# Set page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 8px 16px;
        border-radius: 4px 4px 0px 0px;
        color: #0D47A1 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0D47A1 !important;
        color: white !important;
    }
    h1, h2, h3 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        color: #333333 !important;
        font-size: 16px !important;
        margin-bottom: 5px !important;
    }
    .metric-card h2 {
        color: #0D47A1 !important;
        font-size: 28px !important;
        margin-top: 0 !important;
    }
    .info-box {
        background-color: #e6f2ff;
        border-left: 4px solid #4e8cff;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
        color: #333333; /* Adding a dark text color for better readability */
    }
    /* Button styling - dark blue buttons */
    .stButton > button {
        background-color: #0D47A1 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #1565C0 !important;
        box-shadow: 0 4px 8px rgba(21, 101, 192, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 4px rgba(21, 101, 192, 0.3) !important;
    }
    /* Form submit button */
    .stFormSubmit > button {
        background-color: #0D47A1 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stFormSubmit > button:hover {
        background-color: #1565C0 !important;
        box-shadow: 0 4px 8px rgba(21, 101, 192, 0.3) !important;
    }
    /* Header with animation styling */
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .header-text {
        margin-right: 20px;
    }
    .header-animation {
        height: 200px;
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Function to encode the GIF file
def get_img_with_href(img_path, width=150):
    with open(img_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode()
    return f'<img src="data:image/gif;base64,{b64_img}" width="{width}">'

# Header with animation
st.markdown(
    f"""
    <div class="header-container">
        <div class="header-text">
            <h1>Portfolio Optimization Dashboard</h1>
            <h3>Comparing your portfolio with PSO optimized weights</h3>
        </div>
        <div class="header-animation">
            {get_img_with_href("ParticleSwarmArrowsAnimation.gif", width=300)}
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Session state for storing data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
if 'user_portfolio' not in st.session_state:
    st.session_state.user_portfolio = pd.DataFrame(columns=['Asset', 'Weight'])
if 'available_stocks' not in st.session_state:
    st.session_state.available_stocks = []
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'daily_returns' not in st.session_state:
    st.session_state.daily_returns = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None


# Sidebar configuration
st.sidebar.title("Configuration")

# Data loading section
with st.sidebar.expander("Data Settings", expanded=True):
    data_dir = st.text_input("Data Directory", value="5m")
    n_stocks = st.slider("Max Number of Stocks to Load", 
                         min_value=10, max_value=100, value=30, step=5)
    
    if st.button("Load Stock Data"):
        with st.spinner("Loading stock data..."):
            data_processor = DataProcessor(data_dir)
            st.session_state.stock_data = data_processor.load_stock_data(limit=n_stocks)
            st.session_state.daily_returns = data_processor.calculate_daily_returns()
            st.session_state.train_data, st.session_state.test_data = data_processor.split_train_test(test_ratio=0.2)
            st.session_state.available_stocks = list(st.session_state.daily_returns.columns)
            
            # Reset user portfolio when loading new data
            st.session_state.user_portfolio = pd.DataFrame(columns=['Asset', 'Weight'])
            st.session_state.optimization_done = False
            
            st.session_state.data_loaded = True
            st.success(f"Loaded {len(st.session_state.available_stocks)} stocks!")

# Optimization settings
with st.sidebar.expander("Optimization Settings", expanded=True):
    risk_free_rate = st.slider("Annual Risk-Free Rate (%)", 
                              min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100
    
    objective = st.selectbox("Optimization Objective", 
                            options=['sharpe', 'min_risk', 'max_return', 'calmar', 'sortino'],
                            index=0)
    
    n_particles = st.slider("Number of Particles", 
                           min_value=50, max_value=200, value=100, step=10)
    
    iters = st.slider("Number of Iterations", 
                     min_value=50, max_value=300, value=200, step=10)
    
    initial_investment = st.number_input("Initial Investment ($)", 
                                        min_value=1000, value=10000, step=1000)

# Main content
tabs = st.tabs(["Portfolio Construction", "Optimization Results", "Performance Analysis", "Export Report"])

# Portfolio Construction Tab
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Create Your Portfolio")
        
        # Info box explaining the stock universe restriction
        st.markdown("""
        <div class="info-box">
            <strong>Important:</strong> Your portfolio can only include stocks from the loaded dataset. 
            Both your custom portfolio and the PSO optimization will use the same stock universe.
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load stock data from the sidebar first.")
        else:
            # Display the number of available stocks
            st.info(f"Available stocks for portfolio construction: {len(st.session_state.available_stocks)}")
            
            # Portfolio creation form
            with st.form("portfolio_form"):
                selected_stock = st.selectbox("Select Stock", options=st.session_state.available_stocks)
                weight = st.slider("Weight (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5) / 100
                
                submitted = st.form_submit_button("Add Stock")
                if submitted:
                    # Check if weight sum would exceed 100%
                    current_sum = sum(st.session_state.user_portfolio['Weight'].tolist())
                    if current_sum + weight > 1.0:
                        st.error(f"Total weight would exceed 100%. Remaining available weight: {(1-current_sum)*100:.1f}%")
                    else:
                        # Add stock to portfolio or update if already exists
                        if selected_stock in st.session_state.user_portfolio['Asset'].values:
                            idx = st.session_state.user_portfolio.index[st.session_state.user_portfolio['Asset'] == selected_stock].tolist()[0]
                            st.session_state.user_portfolio.at[idx, 'Weight'] = weight
                        else:
                            new_row = pd.DataFrame({'Asset': [selected_stock], 'Weight': [weight]})
                            st.session_state.user_portfolio = pd.concat([st.session_state.user_portfolio, new_row], ignore_index=True)
            
            # Equal weight portfolio button
            if st.button("Create Equal Weight Portfolio"):
                if len(st.session_state.available_stocks) > 0:
                    weight = 1.0 / len(st.session_state.available_stocks)
                    stocks = []
                    weights = []
                    for stock in st.session_state.available_stocks:
                        stocks.append(stock)
                        weights.append(weight)
                    st.session_state.user_portfolio = pd.DataFrame({
                        'Asset': stocks,
                        'Weight': weights
                    })
            
            # Clear portfolio button
            if st.button("Clear Portfolio"):
                st.session_state.user_portfolio = pd.DataFrame(columns=['Asset', 'Weight'])
                st.session_state.optimization_done = False
            
            # Run optimization button
            if len(st.session_state.user_portfolio) > 0:
                if st.button("Run Optimization"):
                    with st.spinner("Optimizing portfolio..."):
                        try:
                            # Verify all stocks in user portfolio are in the available stock list
                            for asset in st.session_state.user_portfolio['Asset']:
                                if asset not in st.session_state.available_stocks:
                                    raise ValueError(f"Stock {asset} is not in the loaded dataset. Please rebuild your portfolio.")
                                
                            # Run PSO optimization
                            pso_optimizer = PSOPortfolioOptimizer(
                                st.session_state.train_data, 
                                risk_free_rate=risk_free_rate, 
                                objective=objective
                            )
                            
                            pso_weights, pso_return, pso_vol, pso_sharpe = pso_optimizer.optimize(
                                n_particles=n_particles, 
                                iters=iters,
                                verbose=False
                            )
                            
                            # Store optimization results in session state
                            st.session_state.pso_weights = pso_weights
                            st.session_state.pso_return = pso_return
                            st.session_state.pso_vol = pso_vol
                            st.session_state.pso_sharpe = pso_sharpe
                            st.session_state.optimization_result = pso_optimizer.optimization_result
                            
                            # Create mapping of assets to weights for lookup
                            weight_map = {}
                            for i, asset in enumerate(st.session_state.train_data.columns):
                                weight_map[asset] = pso_weights[i]
                            st.session_state.weight_map = weight_map
                            
                            # Calculate equity curves
                            user_weights = np.zeros(len(st.session_state.daily_returns.columns))
                            for idx, row in st.session_state.user_portfolio.iterrows():
                                try:
                                    asset_idx = list(st.session_state.daily_returns.columns).index(row['Asset'])
                                    user_weights[asset_idx] = row['Weight']
                                except ValueError:
                                    st.error(f"Stock {row['Asset']} not found in the dataset. Skipping this stock.")
                                    continue
                            
                            # Normalize weights to sum to 1
                            if np.sum(user_weights) > 0:  # Check if we have valid weights
                                user_weights = user_weights / np.sum(user_weights)
                                
                                # Calculate returns and equity curves
                                user_returns = np.sum(st.session_state.daily_returns * user_weights, axis=1)
                                user_cumulative = (1 + user_returns).cumprod()
                                st.session_state.user_equity = initial_investment * user_cumulative
                                
                                pso_returns = np.sum(st.session_state.daily_returns * pso_weights, axis=1)
                                pso_cumulative = (1 + pso_returns).cumprod()
                                st.session_state.pso_equity = initial_investment * pso_cumulative
                                
                                # Calculate performance metrics
                                metrics = ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
                                perf_comparison = pd.DataFrame(index=metrics)
                                
                                for name, equity in [('User Portfolio', st.session_state.user_equity), 
                                                    ('PSO Optimized', st.session_state.pso_equity)]:
                                    returns = equity.pct_change().dropna()
                                    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
                                    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
                                    volatility = returns.std() * np.sqrt(252)
                                    sharpe = (ann_return - risk_free_rate) / volatility
                                    drawdown = (equity / equity.expanding().max() - 1).min()
                                    
                                    perf_comparison[name] = [total_return, ann_return, volatility, sharpe, drawdown]
                                
                                st.session_state.perf_comparison = perf_comparison
                                st.session_state.optimization_done = True
                                
                                st.success("Portfolio optimization completed!")
                            else:
                                st.error("No valid weights found in your portfolio. Please add stocks to your portfolio.")
                        except Exception as e:
                            st.error(f"Error during optimization: {str(e)}")
    
    with col2:
        st.header("Your Portfolio")
        
        if len(st.session_state.user_portfolio) > 0:
            # Calculate total portfolio weight
            total_weight = sum(st.session_state.user_portfolio['Weight'].tolist())
            
            # Display total weight
            st.metric("Total Weight", f"{total_weight*100:.1f}%")
            
            # Format the weights as percentages
            display_df = st.session_state.user_portfolio.copy()
            display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
            
            # Show the dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Plot portfolio pie chart if there are stocks
            if not display_df.empty:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(st.session_state.user_portfolio['Weight'], 
                       labels=st.session_state.user_portfolio['Asset'],
                       autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title('Your Portfolio Allocation')
                st.pyplot(fig)
        else:
            st.info("Your portfolio is empty. Add stocks to get started.")

# Optimization Results Tab
with tabs[1]:
    if not st.session_state.data_loaded:
        st.warning("Please load stock data from the sidebar first.")
    elif not st.session_state.optimization_done:
        st.info("Run the portfolio optimization to see results.")
    else:
        st.header("Optimization Results")
        
        # Create two columns for weights comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Portfolio Weights")
            
            # Format user weights for display
            user_weights_df = st.session_state.user_portfolio.copy()
            user_weights_df['Weight'] = user_weights_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(user_weights_df, use_container_width=True)
            
            # Plot user weights
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(user_weights_df['Asset'], st.session_state.user_portfolio['Weight'])
            plt.xticks(rotation=90)
            plt.title('Your Portfolio Weights')
            plt.ylabel('Weight')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("PSO Optimized Weights")
            
            # Create DataFrame for PSO weights
            pso_weights_df = pd.DataFrame({
                'Asset': st.session_state.train_data.columns,
                'Weight': st.session_state.pso_weights
            })
            
            # Sort by weight and filter for top weights
            pso_weights_df = pso_weights_df.sort_values('Weight', ascending=False)
            top_n = min(10, len(pso_weights_df))
            top_weights = pso_weights_df.head(top_n)
            
            # Format PSO weights for display
            top_weights['Weight'] = top_weights['Weight'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(top_weights, use_container_width=True)
            
            # Plot PSO weights
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(pso_weights_df.head(top_n)['Asset'], pso_weights_df.head(top_n)['Weight'])
            plt.xticks(rotation=90)
            plt.title('PSO Optimized Portfolio Weights (Top 10)')
            plt.ylabel('Weight')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Plot PSO convergence
        st.subheader("PSO Convergence")
        fig, ax = plt.subplots(figsize=(10, 6))
        cost_history = st.session_state.optimization_result['cost_history']
        ax.plot(range(len(cost_history)), cost_history)
        ax.set_title('PSO Convergence')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Function Value')
        ax.grid(True)
        st.pyplot(fig)

# Performance Analysis Tab
with tabs[2]:
    if not st.session_state.data_loaded:
        st.warning("Please load stock data from the sidebar first.")
    elif not st.session_state.optimization_done:
        st.info("Run the portfolio optimization to see performance analysis.")
    else:
        st.header("Performance Analysis")
        
        # Equity Curve Comparison
        st.subheader("Equity Curve Comparison")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(st.session_state.user_equity.index, st.session_state.user_equity, 
                label=f'User Portfolio (Final: ${st.session_state.user_equity.iloc[-1]:.2f})')
        
        ax.plot(st.session_state.pso_equity.index, st.session_state.pso_equity,
                label=f'PSO Optimized (Final: ${st.session_state.pso_equity.iloc[-1]:.2f})')
        
        ax.axhline(y=initial_investment, color='r', linestyle='--', alpha=0.3, 
                   label=f"Initial Investment (${initial_investment})")
        
        # Add annotations for final values
        ax.annotate(f'${st.session_state.user_equity.iloc[-1]:.2f}', 
                    xy=(st.session_state.user_equity.index[-1], st.session_state.user_equity.iloc[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10)
                    
        ax.annotate(f'${st.session_state.pso_equity.iloc[-1]:.2f}', 
                    xy=(st.session_state.pso_equity.index[-1], st.session_state.pso_equity.iloc[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10)
        
        ax.set_title(f'Portfolio Equity Curves (Initial: ${initial_investment:,.2f})')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Percentage Change Comparison
        st.subheader("Percentage Change from Initial Investment")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate percentage change
        user_pct = (st.session_state.user_equity / initial_investment - 1) * 100
        pso_pct = (st.session_state.pso_equity / initial_investment - 1) * 100
        
        ax.plot(user_pct.index, user_pct, 
                label=f'User Portfolio (+{user_pct.iloc[-1]:.2f}%)')
        
        ax.plot(pso_pct.index, pso_pct,
                label=f'PSO Optimized (+{pso_pct.iloc[-1]:.2f}%)')
        
        # Add annotations for final percentage changes
        ax.annotate(f'+{user_pct.iloc[-1]:.2f}%', 
                    xy=(user_pct.index[-1], user_pct.iloc[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10)
                    
        ax.annotate(f'+{pso_pct.iloc[-1]:.2f}%', 
                    xy=(pso_pct.index[-1], pso_pct.iloc[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10)
        
        ax.set_title('Portfolio Percentage Change from Initial Investment')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage Change (%)')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Performance Metrics Comparison
        st.subheader("Performance Metrics Comparison")
        
        # Format the metrics table for display
        display_metrics = st.session_state.perf_comparison.copy()
        
        # Format different metrics appropriately
        display_metrics.loc['Total Return'] = display_metrics.loc['Total Return'].apply(lambda x: f"{x*100:.2f}%")
        display_metrics.loc['Annualized Return'] = display_metrics.loc['Annualized Return'].apply(lambda x: f"{x*100:.2f}%")
        display_metrics.loc['Volatility'] = display_metrics.loc['Volatility'].apply(lambda x: f"{x*100:.2f}%")
        display_metrics.loc['Sharpe Ratio'] = display_metrics.loc['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        display_metrics.loc['Max Drawdown'] = display_metrics.loc['Max Drawdown'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(display_metrics, use_container_width=True)
        
        # Calculate improvement ratio
        improvement = st.session_state.perf_comparison.loc['Total Return', 'PSO Optimized'] / st.session_state.perf_comparison.loc['Total Return', 'User Portfolio']
        
        # Show metrics in a more visual way
        st.subheader("Key Improvements")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"<div class='metric-card'><h3>Return Improvement</h3>"
                        f"<h2>{improvement:.2f}x</h2></div>", 
                        unsafe_allow_html=True)
            
        with metric_col2:
            sharpe_improvement = st.session_state.perf_comparison.loc['Sharpe Ratio', 'PSO Optimized'] / st.session_state.perf_comparison.loc['Sharpe Ratio', 'User Portfolio']
            st.markdown(f"<div class='metric-card'><h3>Sharpe Ratio Improvement</h3>"
                        f"<h2>{sharpe_improvement:.2f}x</h2></div>", 
                        unsafe_allow_html=True)
            
        with metric_col3:
            risk_reduction = st.session_state.perf_comparison.loc['Max Drawdown', 'User Portfolio'] / st.session_state.perf_comparison.loc['Max Drawdown', 'PSO Optimized']
            st.markdown(f"<div class='metric-card'><h3>Risk Reduction</h3>"
                        f"<h2>{risk_reduction:.2f}x</h2></div>", 
                        unsafe_allow_html=True)

# Export Report Tab
with tabs[3]:
    st.header("Export Report")
    
    if not st.session_state.optimization_done:
        st.info("Run portfolio optimization first to generate a report.")
    else:
        st.write("Generate a PDF report with your portfolio analysis results.")
        
        # Export button
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Create in-memory PDF
                    pdf_buffer = io.BytesIO()
                    
                    with PdfPages(pdf_buffer) as pdf:
                        # Title page
                        plt.figure(figsize=(11, 8.5))
                        plt.axis('off')
                        plt.text(0.5, 0.5, "Portfolio Optimization Report", 
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=24)
                        plt.text(0.5, 0.45, f"Generated on {datetime.now().strftime('%B %d, %Y')}",
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=14)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                        
                        # Equity Curve
                        plt.figure(figsize=(11, 8.5))
                        plt.plot(st.session_state.user_equity.index, st.session_state.user_equity, 
                                label=f'User Portfolio (Final: ${st.session_state.user_equity.iloc[-1]:.2f})')
                        
                        plt.plot(st.session_state.pso_equity.index, st.session_state.pso_equity,
                                label=f'PSO Optimized (Final: ${st.session_state.pso_equity.iloc[-1]:.2f})')
                        
                        plt.axhline(y=initial_investment, color='r', linestyle='--', alpha=0.3, 
                                label=f"Initial Investment (${initial_investment})")
                        
                        plt.title(f'Portfolio Equity Curves (Initial: ${initial_investment:,.2f})', fontsize=16)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Portfolio Value ($)', fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend(fontsize=10)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                        
                        # Percentage Change
                        plt.figure(figsize=(11, 8.5))
                        plt.plot(user_pct.index, user_pct, 
                                label=f'User Portfolio (+{user_pct.iloc[-1]:.2f}%)')
                        
                        plt.plot(pso_pct.index, pso_pct,
                                label=f'PSO Optimized (+{pso_pct.iloc[-1]:.2f}%)')
                        
                        plt.title('Percentage Change from Initial Investment', fontsize=16)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Percentage Change (%)', fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend(fontsize=10)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                        
                        # Portfolio Weights Comparison
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
                        
                        # User weights
                        ax1.pie(st.session_state.user_portfolio['Weight'], 
                              labels=st.session_state.user_portfolio['Asset'],
                              autopct='%1.1f%%', startangle=90)
                        ax1.axis('equal')
                        ax1.set_title('User Portfolio Allocation', fontsize=14)
                        
                        # PSO weights (top 10)
                        pso_top = pso_weights_df.head(min(10, len(pso_weights_df)))
                        if len(pso_top) < len(pso_weights_df):
                            # Add 'Others' category
                            others_weight = pso_weights_df.iloc[min(10, len(pso_weights_df)):]['Weight'].sum()
                            pso_top = pd.concat([
                                pso_top,
                                pd.DataFrame({'Asset': ['Others'], 'Weight': [others_weight]})
                            ])
                        
                        ax2.pie(pso_top['Weight'], 
                              labels=pso_top['Asset'],
                              autopct='%1.1f%%', startangle=90)
                        ax2.axis('equal')
                        ax2.set_title('PSO Optimized Allocation', fontsize=14)
                        
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                        
                        # Performance Metrics Table
                        plt.figure(figsize=(11, 8.5))
                        plt.axis('off')
                        
                        # Create table data
                        raw_data = st.session_state.perf_comparison.copy()
                        table_data = []
                        table_data.append(['Metric', 'User Portfolio', 'PSO Optimized'])
                        for idx, row in raw_data.iterrows():
                            if idx in ['Total Return', 'Annualized Return', 'Volatility', 'Max Drawdown']:
                                user_val = f"{row['User Portfolio']*100:.2f}%"
                                pso_val = f"{row['PSO Optimized']*100:.2f}%"
                            else:
                                user_val = f"{row['User Portfolio']:.2f}"
                                pso_val = f"{row['PSO Optimized']:.2f}"
                            table_data.append([idx, user_val, pso_val])
                        
                        # Create table
                        table = plt.table(cellText=table_data,
                                        loc='center',
                                        cellLoc='center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(12)
                        table.scale(1, 2)
                        
                        plt.title('Performance Metrics Comparison', fontsize=16, y=0.8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                    
                    # Return the PDF
                    pdf_buffer.seek(0)
                    
                    # Create download link
                    b64 = base64.b64encode(pdf_buffer.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="portfolio_analysis.pdf">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("PDF report generated successfully!")
                
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

# Main execution
if __name__ == "__main__":
    pass  # All the execution is handled by Streamlit