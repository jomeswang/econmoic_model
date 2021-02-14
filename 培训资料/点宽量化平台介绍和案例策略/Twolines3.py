# -*- coding: utf-8 -*-

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
    reg_kdata('day', 1)                         # 注册日频行情数据
    # 回测细节设置
    set_backtest(initial_cash=1e8)              # 初始化设置账户总资金
    # 全局变量定义/参数定义
    context.Tlen = len(context.target_list)     # 标的数量
    context.win = 21                            # 计算所需总数据长度
    
    
    context.long_win = 20                       # 20日均线（长均线）参数
    context.short_win = 5                       # 5日均线（短均线）参数
    
    
"""
三、策略运行逻辑函数
"""

# 数据（行情/仓位）——计算逻辑(指标)——下单交易（无持仓/持多单/持空单）

def on_data(context):
    # 获取注册数据
    ##  全部行情数据获取
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)      # 所有标的的K线行情数据
    if data['close'].isna().any():                                    # 行情数据若存在nan值，则跳过
        return
    ## 从全部行情数据中获取想要的数据
    close = data.close.values.reshape(-1, context.win).astype(float)   # 获取收盘价，并转为ndarray类型的二维数组
     # 仓位数据查询,是一个数组
    positions = context.account().positions['volume_long']            # 获取仓位数据：positions=0，表示无持仓
    positions = context.account().positions['volume_short'] 
    
    
    # 循环交易每一个标的
    for i in range(context.Tlen):
        
        # 逻辑计算，计算均线
        mashort = close[i,-context.short_win:].mean()
        malong = close[i,-context.long_win:].mean()
        
      #  ma = ta.SMA(close[i,:],20)
       # malong = ma[-1]
        
        # 下单交易
        if positions[i] == 0:  # 无持仓
            if mashort > malong:  # 短均线>长均线
                # 多单进场
                order_target_value(account_idx=0, target_idx=i, 
                                   target_value=1e8/context.Tlen, side=1,order_type=2, price=0) # 买入下单
        elif positions[i] > 0:   # 持仓
            if mashort < malong:  # 短均线<长均线
                # 出场
                order_target_value(account_idx=0, target_idx=i, target_value=0, side=1,order_type=2, price=0)
"""
四、策略执行脚本
"""
if __name__ == '__main__':
    # 策略回测函数
    run_backtest(strategy_name='TwoLines3', file_path='.', target_list=get_code_list('hs300')['code'],
                 frequency='day', fre_num=1, begin_date='2019-01-01', end_date='2019-05-01', fq=1)


            
    
    