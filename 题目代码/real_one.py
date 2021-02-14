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


def init(context):
    set_backtest(stock_cost_fee=2.5,initial_cash=10000000)
    #  这里将数据导入因为 使用软件的数据来导入，时间太长
    context.data = pd.read_csv("./30支股票行情.csv")

    #  这里是将指数的数据 导入进来进行涨幅率的搜索
    context.stock_data = pd.read_csv("./大湾区指数行情.csv")
    context.stock_profit_rate = []

    context.data['profit'] = context.data['close'] - context.data['open']
    context.data['profitRate'] = 100 * context.data['profit'] / context.data['open']
    context.data_by = context.data.groupby('code')
    context.profitArray = []


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
    list_copy = [x for x in list_df]
    list_copy.sort(reverse=reverse)

    arr = []
    for i in range(0, len(list_df)):
        a = list_copy.index(list_df[i])
        arr.append(a)
    return arr


def draw_pic(profitArray):
    x_data = [x['time'] for x in profitArray]
    y_axis = [x['profit_rate'] for x in profitArray ]
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


def on_data(context):

    # 获得 字长收益率的 一个
    profit = (context.account().cash['total_asset'].tolist()[0] - context.backtest_setting['initial_cash'])
    profit_rate = profit/context.backtest_setting['initial_cash']
    account_profit ={'profit_rate': 100*profit_rate, 'time': format_time(context.now)}
    context.profitArray.append(account_profit)
    print(context.account().position())

    #  获得 指数的增幅
    stock_index = get_correct_index(context.stock_data['time'].tolist(), context.now)
    stock_start_index = get_correct_index(context.stock_data['time'].tolist(), context.backtest_setting['begin_date'])
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
    week_ols_b = []
    week_ols_k = []
    week_ols_r = []
    week_profit_rate = []

    for i in range(0, 30):
        #  这一步将每支股票的数据给分离出来了
        old_target = context.data_by.get_group(context.target_list[i].lower())

        # 这里要将找到切片的位置
        last_index = get_correct_index(old_target['time'].tolist(), context.now)
        if last_index == -1:
            continue

        num = 5;
        while (last_index-num)<0:
            num -= 1

        #  获得上一个星期的数据
        target = old_target.iloc[last_index-num:last_index]

        #  这里是算它的 线性回归
        X = [x for x in range(1,num+1)]
        Y = target['close'].tolist()
        X = sm.add_constant(X)
        results = sm.OLS(Y, X).fit()
        ols_b, ols_k = results.params
        ols_r = results.rsquared

        #  这里算它一周的涨幅
        profit_rate = 100*(target.iloc[-1]['close'] - target.iloc[0]['open'])/target.iloc[0]['open']

        #  最后在这里用数组储存起来因为最后要形成一个df 来处理数据
        code.append(target.iloc[0]['code'])
        week_ols_b.append(ols_b)
        week_ols_k.append(ols_k)
        week_ols_r.append(ols_r)
        week_profit_rate.append(profit_rate)

        # endFor

    week_data ={
        'code': code,
        'week_ols_b': week_ols_b,
        'week_ols_k': week_ols_k,
        'week_ols_r': week_ols_r,
        'week_profit_rate': week_profit_rate
    }
    week_df = pd.DataFrame(week_data)

    #  增加排序
    week_df['week_ols_k_rank'] = sort_df(week_df,'week_ols_k')
    week_df['week_ols_r_rank'] = sort_df(week_df,'week_ols_r')
    week_df['week_profit_rate_rank'] = sort_df(week_df,'week_profit_rate')
    week_df['all_rank'] = week_df['week_ols_k_rank'] + week_df['week_profit_rate_rank']+week_df['week_ols_r_rank']

    #  这里通过对 all_rank 数据来进行排序
    week_df.sort_values('all_rank', inplace=True)
    week_df.to_csv("../第一题/30只股票的测试{year}-{month}-{day}前一个星期的数据.csv".format(year=context.now.year, month=context.now.month, day=context.now.day))

    trade_code =[x.upper() for x in week_df['code'].tolist()[0:10]]
    print('*'*15)
    print(trade_code)

    #  buy_code 是要买的股票 target_code 是有的股票
    target_code = []
    buy_code = []
    sold_code = []
    if context.account().cash['last_amount'][0]!=0:
        hold_code = context.account().position()['code'].tolist()
    else:
        hold_code = []

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