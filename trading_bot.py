# trading_bot.py - Production ML Trading Bot
# ==============================================================================

import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path

# Import our improved strategy
from improved_strategy import ImprovedMLStrategySelector, run_enhanced_backtest
import backtrader as bt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionTradingBot:
    """
    Production-ready trading bot that:
    1. Runs daily analysis
    2. Sends notifications
    3. Saves results
    4. Handles errors gracefully
    5. Tracks performance over time
    """
    
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize performance tracking
        self.performance_file = self.results_dir / 'performance_history.csv'
        self.load_performance_history()
    
    def load_config(self, config_file):
        """Load configuration from file or environment variables"""
        default_config = {
            'symbols': ['^GSPC', 'SPY', 'QQQ'],  # Multiple symbols to analyze
            'lookback_days': 365 * 2,
            'initial_capital': 100000,
            'risk_per_trade': 0.02,
            'notifications': {
                'telegram': {
                    'enabled': False,
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                },
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': os.getenv('EMAIL_USERNAME'),
                    'password': os.getenv('EMAIL_PASSWORD'),
                    'to_email': os.getenv('EMAIL_TO')
                }
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        else:
            config = default_config
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def load_performance_history(self):
        """Load historical performance data"""
        if self.performance_file.exists():
            self.performance_history = pd.read_csv(self.performance_file)
        else:
            self.performance_history = pd.DataFrame(columns=[
                'date', 'symbol', 'strategy', 'signal', 'price', 'portfolio_value',
                'total_return', 'sharpe_ratio', 'max_drawdown'
            ])
    
    def save_performance_data(self, symbol, results_data):
        """Save performance data to CSV"""
        new_row = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'symbol': symbol,
            'strategy': results_data.get('current_strategy', 'unknown'),
            'signal': results_data.get('current_signal', 0),
            'price': results_data.get('current_price', 0),
            'portfolio_value': results_data.get('portfolio_value', 0),
            'total_return': results_data.get('total_return', 0),
            'sharpe_ratio': results_data.get('sharpe_ratio', 0),
            'max_drawdown': results_data.get('max_drawdown', 0)
        }
        
        self.performance_history = pd.concat([
            self.performance_history, 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history.tail(1000)
        
        # Save to file
        self.performance_history.to_csv(self.performance_file, index=False)
    
    def send_telegram_notification(self, message):
        """Send notification via Telegram"""
        if not self.config['notifications']['telegram']['enabled']:
            return
        
        try:
            bot_token = self.config['notifications']['telegram']['bot_token']
            chat_id = self.config['notifications']['telegram']['chat_id']
            
            if not bot_token or not chat_id:
                logger.warning("Telegram credentials not configured")
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
            else:
                logger.error(f"Failed to send Telegram notification: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def send_email_notification(self, subject, message):
        """Send notification via email"""
        if not self.config['notifications']['email']['enabled']:
            return
        
        try:
            email_config = self.config['notifications']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = email_config['to_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            text = msg.as_string()
            server.sendmail(email_config['username'], email_config['to_email'], text)
            server.quit()
            
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def analyze_symbol(self, symbol):
        """Run analysis for a single symbol"""
        logger.info(f"Analyzing {symbol}...")
        
        try:
            # Download recent data for real-time analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)  # Last 100 days for current analysis
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.error(f"No data available for {symbol}")
                return None
            
            # Run quick backtest to get current signals
            cerebro = bt.Cerebro()
            bt_data = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(bt_data)
            cerebro.addstrategy(ImprovedMLStrategySelector)
            cerebro.broker.setcash(self.config['initial_capital'])
            cerebro.broker.setcommission(commission=0.001)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            
            results = cerebro.run()
            
            # Extract results
            strategy = results[0]
            current_price = float(data['Close'].iloc[-1])
            
            # Get performance metrics
            sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            returns = strategy.analyzers.returns.get_analysis().get('rtot', 0)
            drawdown = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            
            # Get current strategy and signal (simplified)
            current_strategy = getattr(strategy, 'current_strategy', 'unknown')
            
            # Calculate current signal
            signals = strategy.get_strategy_signals() if hasattr(strategy, 'get_strategy_signals') else {}
            current_signal = signals.get(current_strategy, 0) if current_strategy in signals else 0
            
            results_data = {
                'symbol': symbol,
                'current_price': current_price,
                'current_strategy': current_strategy,
                'current_signal': current_signal,
                'portfolio_value': cerebro.broker.getvalue(),
                'total_return': returns,
                'sharpe_ratio': sharpe if sharpe != 'N/A' else 0,
                'max_drawdown': drawdown,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            results_file = self.results_dir / f'{symbol}_latest.json'
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Update performance history
            self.save_performance_data(symbol, results_data)
            
            logger.info(f"Analysis complete for {symbol}")
            return results_data
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def generate_daily_report(self, all_results):
        """Generate comprehensive daily report"""
        report = []
        report.append("📊 **Daily ML Trading Strategy Report**")
        report.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if not all_results:
            report.append("❌ No analysis results available")
            return "\n".join(report)
        
        # Summary statistics
        total_signals = sum(1 for r in all_results if r and r['current_signal'] != 0)
        buy_signals = sum(1 for r in all_results if r and r['current_signal'] == 1)
        sell_signals = sum(1 for r in all_results if r and r['current_signal'] == -1)
        
        report.append("📈 **Summary:**")
        report.append(f"• Symbols analyzed: {len([r for r in all_results if r])}")
        report.append(f"• Total signals: {total_signals}")
        report.append(f"• Buy signals: {buy_signals}")
        report.append(f"• Sell signals: {sell_signals}")
        report.append("")
        
        # Individual symbol analysis
        report.append("🔍 **Individual Analysis:**")
        for result in all_results:
            if not result:
                continue
                
            symbol = result['symbol']
            price = result['current_price']
            strategy = result['current_strategy']
            signal = result['current_signal']
            total_return = result['total_return']
            sharpe = result['sharpe_ratio']
            
            signal_emoji = "🔵" if signal == 0 else ("🟢" if signal == 1 else "🔴")
            signal_text = "HOLD" if signal == 0 else ("BUY" if signal == 1 else "SELL")
            
            report.append(f"{signal_emoji} **{symbol}**: ${price:.2f}")
            report.append(f"   Strategy: {strategy}")
            report.append(f"   Signal: {signal_text}")
            report.append(f"   Return: {total_return:.2%}")
            report.append(f"   Sharpe: {sharpe:.3f}")
            report.append("")
        
        # Performance trends (if historical data available)
        if len(self.performance_history) > 5:
            recent_performance = self.performance_history.tail(5)
            avg_return = recent_performance['total_return'].mean()
            avg_sharpe = recent_performance['sharpe_ratio'].mean()
            
            report.append("📊 **5-Day Trends:**")
            report.append(f"• Average Return: {avg_return:.2%}")
            report.append(f"• Average Sharpe: {avg_sharpe:.3f}")
            report.append("")
        
        report.append("⚠️ **Disclaimer:** This is for educational purposes only. Not financial advice.")
        
        return "\n".join(report)
    
    def run_daily_analysis(self):
        """Main function to run daily analysis"""
        logger.info("Starting daily ML trading analysis...")
        
        try:
            all_results = []
            
            # Analyze each symbol
            for symbol in self.config['symbols']:
                result = self.analyze_symbol(symbol)
                all_results.append(result)
            
            # Generate report
            report = self.generate_daily_report(all_results)
            
            # Save report
            report_file = self.results_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Send notifications
            self.send_telegram_notification(report)
            self.send_email_notification("Daily Trading Analysis", report)
            
            logger.info("Daily analysis completed successfully")
            print(report)  # Print to console for GitHub Actions
            
            return all_results
            
        except Exception as e:
            error_msg = f"Error in daily analysis: {e}"
            logger.error(error_msg)
            
            # Send error notification
            self.send_telegram_notification(f"❌ **Trading Bot Error**\n\n{error_msg}")
            self.send_email_notification("Trading Bot Error", error_msg)
            
            raise

def main():
    """Main entry point"""
    try:
        # Check if it's a trading day (Monday-Friday)
        if datetime.now().weekday() >= 5:  # Saturday=5, Sunday=6
            logger.info("Weekend - skipping analysis")
            return
        
        # Initialize and run bot
        bot = ProductionTradingBot()
        results = bot.run_daily_analysis()
        
        # If running in GitHub Actions, set output
        if os.getenv('GITHUB_ACTIONS'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"analysis_completed=true\n")
                f.write(f"signals_count={len([r for r in results if r and r['current_signal'] != 0])}\n")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main()
