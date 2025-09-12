# api/engine.py

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class AssetType(Enum):
    STOCK = "STOCK"
    OPTION = "OPTION"

class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

@dataclass
class TechnicalIndicator:
    name: str
    period: int
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Condition:
    condition_type: str
    operator: str
    value: Union[float, int]
    threshold: Optional[float] = None
    indicator: Optional[str] = None

@dataclass
class StockTrade:
    action: Action
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None

@dataclass
class OptionLeg:
    strike: float
    option_type: OptionType
    action: Action
    quantity: int
    premium: float = 0.0

@dataclass
class RiskManagement:
    stop_loss: Optional[Dict[str, Any]] = None
    take_profit: Optional[Dict[str, Any]] = None
    max_position_size: float = 0.1
    trailing_stop: Optional[Dict[str, Any]] = None

@dataclass
class Position:
    asset_type: AssetType
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    current_value: float = 0.0
    pnl: float = 0.0
    risk_management: Optional[RiskManagement] = None
    legs: List[OptionLeg] = field(default_factory=list)
    shares: int = 0

@dataclass
class Trade:
    asset_type: AssetType
    entry_date: datetime
    exit_date: Optional[datetime] = None
    symbol: str = "STOCK"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 1
    pnl: float = 0.0
    is_open: bool = True
    risk_management: Optional[RiskManagement] = None
    legs: List[OptionLeg] = field(default_factory=list)
    shares: int = 0

class TechnicalIndicatorCalculator:
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        ema_fast = TechnicalIndicatorCalculator.calculate_ema(data, fast)
        ema_slow = TechnicalIndicatorCalculator.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicatorCalculator.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        sma = TechnicalIndicatorCalculator.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        return {'middle': sma, 'upper': sma + (std * std_dev), 'lower': sma - (std * std_dev)}

class BlackScholesCalculator:
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        if T <= 0:
            return max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
        try:
            from scipy.stats import norm
            import math
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            if option_type == OptionType.CALL:
                price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            else:
                price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            return max(price, 0.01)
        except ImportError:
            import math
            intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
            time_value = max(0.1 * S * math.sqrt(T), 0.01)
            return intrinsic + time_value

class UniversalStrategyParser:
    def __init__(self, strategy_json: Dict[str, Any]):
        self.strategy = strategy_json
        self.nodes = {node['id']: node for node in strategy_json.get('nodes', [])}
        self.edges = strategy_json.get('edges', [])
    
    def parse(self) -> Dict[str, Any]:
        edges_by_source = {}
        edges_by_target = {}
        for edge in self.edges:
            edges_by_source.setdefault(edge['source'], []).append(edge['target'])
            edges_by_target.setdefault(edge['target'], []).append(edge['source'])
        
        indicators = self._parse_indicators()
        
        strategy_groups = []
        processed_nodes = set()
        condition_nodes = [n for n in self.nodes.values() if n['type'] == 'priceCondition']
        
        for condition_node in condition_nodes:
            if condition_node['id'] in processed_nodes:
                continue
            group = self._build_strategy_group(condition_node, edges_by_source, edges_by_target, indicators)
            if group:
                strategy_groups.append(group)
                processed_nodes.add(condition_node['id'])
        
        return {
            'strategy_groups': strategy_groups,
            'indicators': indicators,
            'metadata': {
                'name': self.strategy.get('name', 'Unnamed Strategy'),
                'description': self.strategy.get('description', ''),
            }
        }
    
    def _parse_indicators(self) -> Dict[str, TechnicalIndicator]:
        indicators = {}
        for node in self.nodes.values():
            if node['type'] == 'technicalIndicator':
                indicator = self._parse_technical_indicator(node)
                if indicator:
                    indicators[node['id']] = indicator
        return indicators
    
    def _build_strategy_group(self, condition_node: Dict, edges_by_source: Dict, edges_by_target: Dict, indicators: Dict) -> Optional[Dict]:
        condition = self._parse_price_condition(condition_node)
        if not condition:
            return None
        
        connected_indicators = [source_id for source_id in edges_by_target.get(condition_node['id'], []) if source_id in indicators]
        
        actions = {'stock_trades': [], 'option_legs': []}
        risk_management = None
        
        target_ids = edges_by_source.get(condition_node['id'], [])
        for target_id in target_ids:
            target_node = self.nodes.get(target_id)
            if not target_node:
                continue
            
            if target_node['type'] in ['orderType', 'stockTrade']:
                stock_trade = self._parse_stock_trade(target_node)
                if stock_trade:
                    actions['stock_trades'].append(stock_trade)
            elif target_node['type'] == 'optionLeg':
                option_leg = self._parse_option_leg(target_node)
                if option_leg:
                    actions['option_legs'].append(option_leg)
        
        # Look for risk management connected to the trade actions
        action_node_ids = [n['id'] for n in self.nodes.values() if n['type'] in ['stockTrade', 'orderType', 'optionLeg']]
        for an_id in action_node_ids:
             if an_id in edges_by_target.get(condition_node['id'], []):
                 rm_targets = edges_by_source.get(an_id, [])
                 if rm_targets:
                     risk_management = self._parse_risk_management_from_nodes(rm_targets)


        if not actions['stock_trades'] and not actions['option_legs']:
            return None
        
        return {
            'condition': condition,
            'connected_indicators': connected_indicators,
            'actions': actions,
            'risk_management': risk_management
        }

    def _parse_technical_indicator(self, node: Dict) -> Optional[TechnicalIndicator]:
        data = node.get('data', {})
        return TechnicalIndicator(name=data.get('blockType', ''), period=data.get('period', 14), params={k: v for k, v in data.items() if k not in ['blockType', 'period', 'label']})

    def _parse_price_condition(self, node: Dict) -> Optional[Condition]:
        data = node.get('data', {})
        return Condition(condition_type=data.get('blockType', ''), operator=data.get('operator', 'greater_than'), value=data.get('value', 0), threshold=data.get('threshold', data.get('value', 0)))

    def _parse_stock_trade(self, node: Dict) -> Optional[StockTrade]:
        data = node.get('data', {})
        return StockTrade(action=Action(data.get('blockType', 'BUY')), quantity=data.get('quantity', 100), order_type=OrderType.MARKET, limit_price=data.get('limit_price'))

    def _parse_option_leg(self, node: Dict) -> Optional[OptionLeg]:
        data = node.get('data', {})
        return OptionLeg(strike=data.get('strike', 100), option_type=OptionType(data.get('blockType', 'CALL')), action=Action(data.get('action', 'BUY')), quantity=data.get('quantity', 1))

    def _parse_risk_management_from_nodes(self, target_ids: List[str]) -> Optional[RiskManagement]:
        stop_loss, take_profit = None, None
        for target_id in target_ids:
            target_node = self.nodes.get(target_id)
            if not target_node:
                continue
            if target_node['type'] == 'stopLoss':
                stop_loss = target_node.get('data', {})
            elif target_node['type'] == 'takeProfit':
                take_profit = target_node.get('data', {})
        
        if stop_loss or take_profit:
            return RiskManagement(stop_loss=stop_loss, take_profit=take_profit)
        return None

class HybridBacktester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.bs_calculator = BlackScholesCalculator()
        self.risk_free_rate = 0.02
        self.volatility = 0.25
        self.days_to_expiration = 30
        self.cached_indicators: Dict[str, pd.Series] = {}

    def run_backtest(self, data: pd.DataFrame, parsed_strategy: Dict[str, Any]) -> Dict[str, Any]:
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        self._calculate_indicators(data, parsed_strategy['indicators'])
        
        for i, row in data.iterrows():
            current_date = row['Date']
            current_price = row['Close']
            self._update_positions(current_price, current_date, i)
            self._check_entry_conditions(row, parsed_strategy, current_date, i)
            self._check_exit_conditions(current_price, current_date)
            portfolio_value = self._calculate_portfolio_value()
            self.equity_curve.append(portfolio_value)
            self.dates.append(current_date)
        
        return self._generate_results()

    def _calculate_indicators(self, data: pd.DataFrame, indicators: Dict[str, TechnicalIndicator]):
        for indicator_id, indicator in indicators.items():
            if indicator.name == 'RSI':
                self.cached_indicators[indicator_id] = self.indicator_calculator.calculate_rsi(data['Close'], indicator.period)
            elif indicator.name == 'SMA':
                self.cached_indicators[indicator_id] = self.indicator_calculator.calculate_sma(data['Close'], indicator.period)
            elif indicator.name == 'EMA':
                self.cached_indicators[indicator_id] = self.indicator_calculator.calculate_ema(data['Close'], indicator.period)
            elif indicator.name == 'MACD':
                macd_data = self.indicator_calculator.calculate_macd(data['Close'])
                self.cached_indicators[f"{indicator_id}_macd"] = macd_data['macd']
                self.cached_indicators[f"{indicator_id}_signal"] = macd_data['signal']

    def _update_positions(self, current_price: float, current_date: datetime, data_index: int):
        for position in self.positions.values():
            if position.asset_type == AssetType.STOCK:
                position.current_value = position.shares * current_price
                position.pnl = (position.current_value - position.entry_price) if position.shares > 0 else (position.entry_price - position.current_value)
            elif position.asset_type == AssetType.OPTION:
                position.current_value = self._calculate_options_position_value(position, current_price, current_date)
                position.pnl = position.current_value - position.entry_price

    def _calculate_options_position_value(self, position: Position, current_price: float, current_date: datetime) -> float:
        total_value = 0.0
        days_elapsed = (current_date - position.entry_date).days
        time_to_expiration = max((self.days_to_expiration - days_elapsed) / 365, 0.001)
        for leg in position.legs:
            option_price = self.bs_calculator.calculate_option_price(S=current_price, K=leg.strike, T=time_to_expiration, r=self.risk_free_rate, sigma=self.volatility, option_type=leg.option_type)
            leg_value = option_price * leg.quantity * 100
            total_value += leg_value if leg.action == Action.BUY else -leg_value
        return total_value * position.quantity

    def _check_entry_conditions(self, row: pd.Series, parsed_strategy: Dict[str, Any], current_date: datetime, data_index: int):
        current_price = row['Close']
        for group in parsed_strategy['strategy_groups']:
            if self._evaluate_condition(group['condition'], group['connected_indicators'], current_price, data_index):
                self._execute_strategy_group(group, current_price, current_date)

    def _evaluate_condition(self, condition: Condition, connected_indicators: List[str], current_price: float, data_index: int) -> bool:
        if not connected_indicators:
            return self._check_condition_logic(condition, current_price, None)
        if len(connected_indicators) == 1:
            indicator_id = connected_indicators[0]
            if indicator_id in self.cached_indicators:
                indicator_values = self.cached_indicators[indicator_id]
                if data_index < len(indicator_values):
                    indicator_value = indicator_values.iloc[data_index]
                    if pd.notna(indicator_value):
                        return self._check_condition_logic(condition, indicator_value, condition.threshold)
            return False
        elif len(connected_indicators) >= 2:
            id1, id2 = connected_indicators[0], connected_indicators[1]
            if id1 in self.cached_indicators and id2 in self.cached_indicators:
                values1 = self.cached_indicators[id1]
                values2 = self.cached_indicators[id2]
                if data_index < len(values1) and data_index < len(values2):
                    val1, val2 = values1.iloc[data_index], values2.iloc[data_index]
                    if pd.notna(val1) and pd.notna(val2):
                        return self._check_condition_logic(condition, val1, val2)
            return False
        return False

    def _check_condition_logic(self, condition: Condition, value1: float, value2: Optional[float]) -> bool:
        compare_value = value2 if value2 is not None else condition.threshold
        if condition.condition_type == 'LessThan':
            return value1 < compare_value
        elif condition.condition_type == 'GreaterThan':
            return value1 > compare_value
        return False

    def _execute_strategy_group(self, group: Dict, current_price: float, current_date: datetime):
        actions = group['actions']
        risk_management = group.get('risk_management')
        for stock_trade in actions['stock_trades']:
            self._execute_stock_trade(stock_trade, current_price, current_date, risk_management)
        if actions['option_legs']:
            self._execute_options_trade(actions['option_legs'], current_price, current_date, risk_management)

    def _execute_stock_trade(self, stock_trade: StockTrade, current_price: float, entry_date: datetime, risk_management: Optional[RiskManagement]):
        position_value = stock_trade.quantity * current_price
        if position_value > self.current_capital:
            return
        
        position_id = f"stock_{entry_date.strftime('%Y%m%d_%H%M%S')}"
        position = Position(asset_type=AssetType.STOCK, symbol="STOCK", entry_date=entry_date, entry_price=position_value, quantity=1, shares=stock_trade.quantity, current_value=position_value, risk_management=risk_management)
        self.positions[position_id] = position
        self.current_capital -= position_value
        self.trades.append(Trade(asset_type=AssetType.STOCK, entry_date=entry_date, symbol="STOCK", entry_price=current_price, shares=stock_trade.quantity, quantity=1, risk_management=risk_management))

    def _execute_options_trade(self, option_legs: List[OptionLeg], current_price: float, entry_date: datetime, risk_management: Optional[RiskManagement]):
        net_premium = 0.0
        time_to_expiration = self.days_to_expiration / 365
        priced_legs = []
        for leg in option_legs:
            premium = self.bs_calculator.calculate_option_price(S=current_price, K=leg.strike, T=time_to_expiration, r=self.risk_free_rate, sigma=self.volatility, option_type=leg.option_type)
            leg.premium = premium
            priced_legs.append(leg)
            net_premium += -premium * leg.quantity * 100 if leg.action == Action.BUY else premium * leg.quantity * 100

        if abs(net_premium) > self.current_capital:
            return
        position_id = f"options_{entry_date.strftime('%Y%m%d_%H%M%S')}"
        position = Position(asset_type=AssetType.OPTION, symbol="OPTIONS", entry_date=entry_date, entry_price=net_premium, quantity=1, legs=priced_legs, current_value=net_premium, risk_management=risk_management)
        self.positions[position_id] = position
        self.current_capital += net_premium
        self.trades.append(Trade(asset_type=AssetType.OPTION, entry_date=entry_date, symbol="OPTIONS", entry_price=net_premium, legs=priced_legs, quantity=1, risk_management=risk_management))
        
    def _check_exit_conditions(self, current_price: float, current_date: datetime):
        positions_to_close = []
        for pos_id, position in self.positions.items():
            should_exit = False
            pnl_percent = position.pnl / abs(position.entry_price) if position.entry_price != 0 else 0
            
            days_held = (current_date - position.entry_date).days
            max_days = 30 if position.asset_type == AssetType.OPTION else 60
            if days_held >= max_days:
                should_exit = True
            
            if position.risk_management:
                if position.risk_management.stop_loss and pnl_percent <= -abs(position.risk_management.stop_loss.get('value', 0.05)):
                    should_exit = True
                if position.risk_management.take_profit and pnl_percent >= abs(position.risk_management.take_profit.get('value', 0.1)):
                    should_exit = True
            
            if should_exit:
                positions_to_close.append(pos_id)
        
        for pos_id in positions_to_close:
            self._close_position(pos_id, current_price, current_date)

    def _close_position(self, position_id: str, exit_price: float, exit_date: datetime):
        if position_id not in self.positions:
            return
        position = self.positions.pop(position_id)
        
        final_value = position.shares * exit_price if position.asset_type == AssetType.STOCK else position.current_value
        
        for trade in reversed(self.trades):
            if trade.entry_date == position.entry_date and trade.is_open:
                trade.exit_date = exit_date
                trade.exit_price = exit_price if position.asset_type == AssetType.STOCK else final_value
                trade.pnl = (exit_price - trade.entry_price) * trade.shares if position.asset_type == AssetType.STOCK else final_value - trade.entry_price
                trade.is_open = False
                break
        
        if position.asset_type == AssetType.STOCK:
            self.current_capital += final_value
        elif position.asset_type == AssetType.OPTION:
            self.current_capital += final_value

    def _calculate_portfolio_value(self) -> float:
        total_positions_value = sum(pos.current_value for pos in self.positions.values())
        return self.current_capital + total_positions_value
    def suggest_optimizations(self, all_trades: List[Trade], metrics: Dict) -> List[str]:
        """Provides suggestions for optimizing the strategy based on backtest metrics."""
        suggestions = []
        
        # Rule 1: Check for profitability
        if metrics['total_return_pct'] < 0:
            suggestions.append("The strategy was not profitable. Consider adjusting your entry or exit conditions.")

        # Rule 2: Check for a low win rate, which might indicate poor entry signals
        if metrics['win_rate_pct'] < 40.0 and metrics['total_trades'] > 10:
            suggestions.append("A low win rate suggests that entry signals may be unreliable. Try adding a second indicator as a confirmation signal.")
        
        # Rule 3: Check for high drawdown, suggesting poor risk management
        if metrics['max_drawdown_pct'] < -10.0:
            # Check if a stop-loss was used at all (this logic is simplified)
            has_risk_management = any(t.risk_management for t in all_trades)
            if not has_risk_management:
                suggestions.append("Your strategy has a high maximum drawdown. Implementing a stop-loss order could significantly reduce risk.")
            else:
                suggestions.append("The current risk management settings may not be effective. Consider adjusting your stop-loss or take-profit levels.")

        # Rule 4: Check for a high number of trades, which can lead to over-trading and high fees
        if metrics['total_trades'] > 50:
            suggestions.append("The strategy generated a high number of trades. Using a longer period for your indicators could reduce noise and trading costs.")
        
        if not suggestions:
            suggestions.append("The strategy seems reasonable. You could try optimizing indicator periods for better results.")
            
        return suggestions
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        # Close any remaining open positions
        final_date = self.dates[-1] if self.dates else datetime.now()
        
        # Get final price for closing remaining positions
        final_price = self.equity_curve[-1] / len(self.positions) if self.positions and self.equity_curve else 100
        
        for pos_id in list(self.positions.keys()):
            position = self.positions[pos_id]
            if position.asset_type == AssetType.STOCK:
                self._close_position(pos_id, final_price, final_date)
            else:
                self._close_position(pos_id, final_date)
        
        # Convert trades to serializable format
        trades_as_dicts = []
        for trade in self.trades:
            trade_dict = {
                'asset_type': trade.asset_type.value,
                'entry_date': trade.entry_date.isoformat(),
                'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
                'symbol': trade.symbol,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'shares': trade.shares,
                'pnl': trade.pnl,
                'is_open': trade.is_open
            }
            trades_as_dicts.append(trade_dict)
        
        # Calculate metrics
        equity_series = pd.Series(self.equity_curve, index=self.dates) if self.dates and self.equity_curve else pd.Series([self.initial_capital])
        returns = equity_series.pct_change().dropna() if len(equity_series) > 1 else pd.Series([0])
        metrics = self._calculate_metrics(self.trades, equity_series, returns)
        
        # NEW: Generate optimization suggestions
        suggestions = self.suggest_optimizations(self.trades, metrics)
        
        return {
            'metrics': metrics,
            'equity_curve': list(zip([d.isoformat() for d in self.dates], self.equity_curve)) if self.dates else [],
            'trades': trades_as_dicts,
            'suggestions': suggestions # Add suggestions to the results dictionary
        }


    def _calculate_metrics(self, all_trades: List[Trade], equity_series: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve or len(self.equity_curve) == 0:
            return {
                'total_return_pct': 0.0,
                'final_capital': float(self.initial_capital),
                'total_trades': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
            }
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        closed_trades = [t for t in all_trades if not t.is_open]
        
        if not closed_trades:
            win_rate, profit_factor = 0.0, 0.0
        else:
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = 999999.0  # Large number instead of inf
            else:
                profit_factor = 0.0

        # Calculate drawdown
        if len(equity_series) > 1:
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0

        # Calculate Sharpe ratio
        if len(returns) > 1 and returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        # Ensure all values are finite and JSON-serializable
        import math
        
        def safe_float(value):
            if math.isinf(value):
                return 999999.0 if value > 0 else -999999.0
            elif math.isnan(value):
                return 0.0
            return float(value)

        return {
            'total_return_pct': safe_float(total_return * 100),
            'final_capital': safe_float(self.equity_curve[-1]),
            'total_trades': int(len(closed_trades)),
            'win_rate_pct': safe_float(win_rate * 100),
            'profit_factor': safe_float(profit_factor),
            'sharpe_ratio': safe_float(sharpe_ratio),
            'max_drawdown_pct': safe_float(max_drawdown * 100),
        }

class HybridVisualizationEngine:
    @staticmethod
    def create_single_interactive_plot(backtest_results: Dict, benchmark_data: pd.DataFrame) -> str:
        """
        Creates a single interactive plot with Equity Curve and Drawdown.
        Returns the plot as an HTML div string.
        """
        try:
            if not backtest_results.get('equity_curve'):
                return "<div>No equity curve data available.</div>"

            # Prepare data
            equity_data = backtest_results['equity_curve']
            
            # Handle both tuple format (date_str, value) and dict format
            if isinstance(equity_data[0], tuple):
                dates = [datetime.fromisoformat(d) if isinstance(d, str) else d for d, v in equity_data]
                equity = [v for d, v in equity_data]
            else:
                # If it's not tuples, assume it's a list of values with corresponding dates
                dates = [datetime.fromisoformat(d) for d in backtest_results.get('dates', [])]
                equity = equity_data
            
            if not dates or not equity:
                return "<div>No valid date/equity data found.</div>"

            equity_series = pd.Series(equity, index=dates)

            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]
            )

            # 1. Equity Curve
            fig.add_trace(
                go.Scatter(
                    x=dates, y=equity, name='Strategy',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )

            # Add Benchmark if valid data exists
            try:
                if benchmark_data is not None and not benchmark_data.empty:
                    benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
                    
                    # Filter benchmark data to match backtest period
                    start_date = min(dates)
                    end_date = max(dates)
                    benchmark_filtered = benchmark_data[
                        (benchmark_data['Date'] >= start_date) & 
                        (benchmark_data['Date'] <= end_date)
                    ].copy()
                    
                    if not benchmark_filtered.empty:
                        initial_price = benchmark_filtered['Close'].iloc[0]
                        initial_capital = equity[0]  # Use first equity value as initial capital
                        benchmark_equity = (benchmark_filtered['Close'] / initial_price) * initial_capital
                        
                        fig.add_trace(
                            go.Scatter(
                                x=benchmark_filtered['Date'], y=benchmark_equity, 
                                name='Buy & Hold',
                                line=dict(color='#ff7f0e', width=1.5, dash='dash')
                            ),
                            row=1, col=1
                        )
            except Exception as e:
                print(f"Warning: Could not add benchmark: {e}")
            
            # 2. Drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            fig.add_trace(
                go.Scatter(
                    x=dates, y=drawdown, name='Drawdown',
                    fill='tozeroy', fillcolor='rgba(239, 83, 80, 0.3)',
                    line=dict(color='rgba(239, 83, 80, 0.7)', width=1)
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Strategy Performance: Equity Curve & Drawdown',
                template='plotly_white',
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=20, t=80, b=20),
                yaxis1_title='Portfolio Value ($)',
                yaxis2_title='Drawdown (%)',
                xaxis2_title='Date',
                yaxis1_tickformat='$,.0f',
                yaxis2_ticksuffix='%'
            )

            return fig.to_html(full_html=False, include_plotlyjs=False)
            
        except Exception as e:
            print(f"Plot generation error: {str(e)}")
            return f"<div>Plot generation failed: {str(e)}</div>"