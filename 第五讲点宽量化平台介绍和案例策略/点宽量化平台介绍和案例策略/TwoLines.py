
"""
一、工具包导入
"""
from atrader import *
import numpy as np
"""
二、初始化
"""
def init(context):
    # 注册数据
    reg_kdata('day', 1)                        # 注册日频行情数据
    # 回测细节设置
    set_backtest(initial_cash=1e8)            # 初始化设置账户总资金
    # 全局变量定义/参数定义
    context.win = 21                            # 计算所需总数据长度
    context.long_win = 20                       # 20日均线（长均线）参数
    context.short_win = 5                       # 5日均线（短均线）参数
    context.Tlen = len(context.target_list)     # 标的数量
"""
三、策略运行逻辑函数
"""
def on_data(context):
    # 获取注册数据
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)      # 所有标的的K线行情数据
    if data['close'].isna().any():                                    # 行情数据若存在nan值，则跳过
        return
    close = data.close.values.reshape(-1, context.win).astype(float)   # 获取收盘价，并转为ndarray类型的二维数组
    # 仓位数据查询
    positions = context.account().positions['volume_long'].values    # 获取仓位数据：positions=0，表示无持仓
    # 逻辑计算
    mashort = close[:, -5:].mean(axis=1)                     # 短均线：5日均线
    malong = close[:, -20:].mean(axis=1)                     # 长均线：20日均线
   
    target = np.array(range(context.Tlen))                   # 获取标的序号
    long = np.logical_and(positions == 0, mashort > malong)     # 未持仓，且短均线上穿长均线为买入信号
    short = np.logical_and(positions > 0, mashort < malong)     # 持仓，且短均线下穿长均线为卖出信号
   
    target_long = target[long].tolist()                      # 获取买入信号标的的序号
    target_short = target[short].tolist()                    # 获取卖出信号标的的序号
    # 策略下单交易：
    for targets in target_long:
        order_target_value(account_idx=0, target_idx=targets, target_value=1e8/context.Tlen, side=1,order_type=2, price=0) # 买入下单
    for targets in target_short:
        order_target_volume(account_idx=0, target_idx=targets, target_volume=0, side=1,order_type=2, price=0)              # 卖出平仓
"""
四、策略执行脚本
"""
if __name__ == '__main__':
    # 策略回测函数
    run_backtest(strategy_name='TwoLines', file_path='TwoLines.py', target_list=get_code_list('hs300')['code'],
                 frequency='day', fre_num=1, begin_date='2019-01-01', end_date='2019-05-01', fq=1)




