#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------------
    @Author  : 罗立旺
    @Time    : 2020/11/2 22:38
    @Software: PyCharm
----------------------------------------
"""

from atrader import *
import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
import datetime
import pyecharts.options as opts
from pyecharts.charts import Line
import pyecharts.options as opts
from pyecharts.charts import Line
import alphalens as al


def init(context):
    set_backtest(stock_cost_fee=2.5,initial_cash=10000000)
    #  这里将数据导入因为 使用软件的数据来导入，时间太长
    context.data = pd.read_csv("./30支股票行情.csv")
    reg_factor([ 'bp', 'volatility', 'roe', 'roa', 'bias6'])
    context.factor_list = ['bp', 'volatility', 'roe', 'roa', 'bias6']
    #  这里是将指数的数据 导入进来进行涨幅率的搜索
    context.stock_data = pd.read_csv("./大湾区指数行情.csv")
    context.stock_profit_rate = []

    context.data['profit'] = context.data['close'] - context.data['open']
    context.data['profitRate'] = 100 * context.data['profit'] / context.data['open']
    context.data_by = context.data.groupby('code')
    context.profitArray = []
    context.factor_data = [get_data(context, x) for x in context.factor_list]

    print(context.factor_data)
    # print(context.factor_data.tolist())
    # 注册 K 线数据 为 1 天
    # reg_kdata('day', 1)


def judge_time(time1, time2):
    if time1.year == time2.year and time1.month == time2.month and time1.day == time2.day:
        return 1
    return 0


def change_time(time):
    return datetime.datetime.strptime(time, "%Y/%m/%d %H:%M")


def get_correct_index(data_list, time):
    for i in range(0, len(data_list)):
        time1 = datetime.datetime.strptime(data_list[i], "%Y/%m/%d %H:%M")
        if judge_time(time1, time):
            return i
    return -1


def format_time(time):
    return "{year}-{month}-{day}".format(year=time.year, month=time.month, day=time.day)


def sort_df(df, name,reverse=True):
    # reverse  False 升序
    list_df = df[name].tolist()
    # print(list_df)
    # print('*'*12)
    list_copy = [x for x in list_df]
    list_copy.sort(reverse=reverse)
    # print(list_copy)
    # print(list_df)

    arr = []
    for i in range(0, len(list_df)):
        a = list_copy.index(list_df[i])
        arr.append(a)
        # print(a)
    # a = [list_copy.index(list_df[i]) for i in range(0, len(list_df))]
    # print(a)

    return arr


def draw_pic(profitArray):
    x_data = [x['time'] for x in profitArray]
    y_axis = [x['profit_rate'] for x in profitArray ]
    # print(x_data)
    # print('*'*15)
    # print(y_axis)
    # print(profitArray)
    (
        Line()
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="收益率",
            stack="总量",
            y_axis=y_axis,
            label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="策略收益率图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
            .render("策略收益率.html")
    )


def draw_two(stock_profit, profit):
    x_data = [x['time'] for x in profit ]
    #  策略收益
    y_axis = [x['profit_rate'] for x in profit]
    #  湾区涨幅
    y_stock_axis = [x['profit_rate'] for x in stock_profit]
    c = (
        Line()
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis("策略收益", y_axis, is_smooth=True)
            .add_yaxis("大湾区指数", y_stock_axis, is_smooth=True)
            .set_series_opts(
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="湾区指数和策略对比图"),
            xaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                is_scale=False,
                boundary_gap=False,
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
            .render("湾区指数和策略对比图.html")
    )


def get_data(context, factor_name):
    # def get_max_ir():

    begin_date = context.backtest_setting['begin_date']
    end_date = context.backtest_setting['end_date']

    df = pd.DataFrame()
    #   在这里获得 因子 数据
    for i in range(len(context.target_list)):
        factor_data = get_factor_by_code(factor_list=[factor_name], target=context.target_list[i],
                                         begin_date=begin_date,
                                         end_date=end_date)
        data = get_kdata(target_list=context.target_list[i], frequency='day', fre_num=1,
                         begin_date=begin_date, end_date=end_date, fq=1, fill_up=True, df=True,
                         sort_by_date=False)
        data = data.join(factor_data)
        df = df.append(data)
    # 因子格式化
    factor = df.groupby(['date', 'code'])[factor_name].sum()
    # factor.to_csv("./test.csv")
    #  这里收盘价的格式化
    data = df.pivot(index="date", columns="code", values="close")
    a = al.utils.get_clean_factor_and_forward_returns(factor, data, max_loss=1)
    _return = al.performance.factor_returns(a).mean()
    #  IC  值分天
    ic_date = al.performance.factor_information_coefficient(a)
    #  IC 值 取平均
    ic_mean = al.performance.mean_information_coefficient(a)
    # print(ic_mean)
    # print(ic_mean.tolist())
    return  ic_mean.tolist()[1]


#  处理缺失值函数
def fill_ndarray(t1):
    # print(t1)
    for i in range(t1.shape[1]):  # 遍历每一列（每一列中的nan替换成该列的均值）
        temp_col = t1[:, i]  # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有nan
            temp_not_nan_col = temp_col[temp_col == temp_col]  # 去掉nan的ndarray

            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  # mean()表示求均值。
    return t1


def on_data(context):
    # 获得 字长收益率的 一个
    # print(context.backtest_setting)
    # print(context.account().cash)
    profit = (context.account().cash['total_asset'].tolist()[0] - context.backtest_setting['initial_cash'])
    # print(profit)
    profit_rate = profit/context.backtest_setting['initial_cash']
    account_profit ={'profit_rate': 100*profit_rate, 'time': format_time(context.now)}
    context.profitArray.append(account_profit)
    print(context.account().position())

    #  获得 指数的增幅
    stock_index = get_correct_index(context.stock_data['time'].tolist(), context.now)
    stock_start_index = get_correct_index(context.stock_data['time'].tolist(), context.backtest_setting['begin_date'])
    print(stock_start_index,stock_index)
    print('*'*15)
    if stock_index != -1:
        stock_rate = (context.stock_data.iloc[stock_index]['close'] - context.stock_data.iloc[stock_start_index]['close'])/context.stock_data.iloc[stock_start_index]['close']
        context.stock_profit_rate.append({'profit_rate': 100*stock_rate, 'time': format_time(context.now)})

    # 这里是判断是否是最后一天 如果是的话就将 图画出来
    if judge_time(context.backtest_setting['end_date'], context.now):
        draw_two(context.stock_profit_rate, context.profitArray)
        draw_pic(context.profitArray)

    #  判断是不是星期一
    if context.now.weekday() != 0:
        return
    code = []

    factor = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=1, df=True)
    fector_day = []
    for i in range(len(context.factor_list)):
        # print(factor[factor['factor'] == context.factor_list[i]])
        tmp = [x for x in factor[factor['factor'] == context.factor_list[i]]['value'].tolist()]
        # print(tmp)
        fector_day.append(tmp)
    #     1*6
    context.factor_data = np.mat(context.factor_data)
    #  6*30
    fector_day = fill_ndarray(np.mat(fector_day))
    # print(context.factor_data, fector_day)
    rate = context.factor_data * fector_day

    # endFor

    week_data ={
        'code': context.target_list,
        'fector':rate[0].tolist()[0]
    }
    print(week_data)
    week_df = pd.DataFrame(week_data)

    #  增加排序
    week_df['factor_rank'] = sort_df(week_df, 'fector')

    #  这里通过对 factor_rank 数据来进行排序
    week_df.sort_values('factor_rank', inplace=True)

    trade_code =[x.upper() for x in week_df['code'].tolist()[0:10]]
    print('*'*15)
    print(trade_code)

    #  buy_code 是要买的股票 target_code 是有的股票
    target_code = []
    buy_code = []
    sold_code = []
    # print(context.account().cash['last_amount'].tolist())
    if context.account().cash['last_amount'][0]!=0:
        hold_code = context.account().position()['code'].tolist()
    else:
        hold_code = []

    # context.account().position()
    #  在这里将 已持仓的 继续持仓，然后卖出剩下的  然后买入选入持仓池里面的
    for i in range(0,10):
        if trade_code[i] in hold_code:
            target_code.append(trade_code[i])
        else:
            buy_code.append(trade_code[i])
        if len(hold_code)!=0  and i <len(hold_code) and not(hold_code[i] in trade_code):
            sold_code.append(hold_code[i])
            # 对索引号为0的账户中索引号为0的标的，调仓到多头100股，委托方式为市价委托
            order_target_volume(account_idx=0, target_idx=context.target_list.index(hold_code[i]), target_volume=0, side=1, order_type=2, price=0.0)
    print('hold_code:', hold_code)
    print('buy_code', buy_code)
    print('sold_code', sold_code)


    # 检测是否持仓 如果持仓 那么 平仓
    # order_close_all(account_idx=0)
    # order_cancel_all(account_indice=[0])

    #  在这里进行交易
    for i in range(0,len(buy_code)):
        # 对索引号为0的账户开仓买入10000元价值，索引号为0的标的，，委托方式为市价委托
        every_money = context.backtest_setting['initial_cash']*0.1
        all_money = context.account().cash['total_asset'].tolist()[0] -len(target_code) * every_money

        print(all_money-i*every_money,'*\n',every_money)

        if all_money-i*every_money >= every_money:
            print("+"*15)
            order_value(account_idx=0, target_idx=context.target_list.index(buy_code[i]), value=every_money, side=1, position_effect=1, order_type=2, price=0.0)
        elif all_money-i*every_money >0:
            print('*'*15)
            order_percent(account_idx=0, target_idx=context.target_list.index(buy_code[i]), percent=1, side=1, position_effect=1, order_type=2, price=0.0)


if __name__ == '__main__':
    begin_date = "2011-01-04"
    end_date = "2020-10-30"

    #  获取 粤港澳 大湾区 成分股信息
    targetList = get_code_list('szse.399999')['code'].tolist()
    # print(targetList, len(targetList))
    run_backtest(
        strategy_name= "测试第一题",
        file_path=".",
        target_list=targetList,
        frequency='day',
        fre_num=1,
        begin_date=begin_date,
        end_date=end_date,
        fq=1
    )