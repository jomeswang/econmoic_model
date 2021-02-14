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
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats


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


def get_distribution(df, index):
    # index 为1 时说明 获取夏普率最高的组合， 2 时获取方差最小的组合

    #  这一步是 数据的处理
    df_group = df.groupby('code')
    code_list = df['code'].unique()
    data_df = {}
    data_time = []
    # df =
    for i in range(0, len(code_list)):
        code_by = df_group.get_group(code_list[i])
        close_data = code_by['close'].tolist()[0]
        time_data = code_by['time'].tolist()[0]
        code_dist_data = {}
        for j in range(0, len(close_data)):
            code_dist_data[time_data[j]] = float(close_data[j])
        data_df[code_list[i]] = code_dist_data

    data = pd.DataFrame(data_df).dropna()

    # #计算对数收益率。金融计算收益率的时候大部分用对数收益率 (Log Return) 而不是用算数收益率
    log_returns = np.log( data.pct_change() + 1)
    log_returns = log_returns.dropna()
    # #使用对数收益率为收益率
    rets = log_returns

    def statistics(weights):
        # 根据权重，计算资产组合收益率/波动率/夏普率。
        # 输入参数
        # ==========
        # weights : array-like 权重数组
        # 权重为股票组合中不同股票的权重
        # 返回值
        # =======
        # pret : float
        #      投资组合收益率
        # pvol : float
        #      投资组合波动率
        # pret / pvol : float
        #    夏普率，为组合收益率除以波动率，此处不涉及无风险收益率资产

        weights = np.array(weights)
        pret = np.sum(rets.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))+1e-2)
        # print( "权重",weights,"波动率:",pret,'\n',"夏普比率:",pvol )

        return np.array([pret, pvol, pret / pvol])

    # 则我们的最大化夏普率问题可以转变为最小化负的夏普率问题。定义输入权重分配，返回负夏普率的函数为：
    def min_func_sharpe(weights):
        # print(statistics(weights)[2])
        return -statistics(weights)[2]

    # 如果我们想知道最小方差的投资组合
    def min_func_variance(weights):
        return statistics(weights)[1] ** 2

    # 输入权重，输出波动率的函数为
    def min_func_port(weights):
        return statistics(weights)[1]

    number_of_assets = len(code_list)

    # 即每个权重需要在0到1之间
    bnds = tuple((0, 1) for x in range(number_of_assets))
    # 即权重之和为1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})


    if index == 1:
        #  获得最大夏普率
        opts = sco.minimize(min_func_sharpe, number_of_assets * [1. / number_of_assets], method='SLSQP', bounds=bnds,
                            constraints=cons, options={'maxiter': 100, 'disp': True, 'ftol': 1e-6})
        print("仓位分配：", opts['x'].round(3))
        print(statistics(opts['x'].round(3))[0], statistics(opts['x'].round(3))[1],statistics(opts['x'].round(3))[2])
        return opts['x'].round(3)

    else:
        # 方差最小
        optv = sco.minimize(min_func_variance, number_of_assets * [1. / number_of_assets], method='SLSQP', bounds=bnds,
                            constraints=cons,  options={'maxiter': 100, 'disp': True, 'ftol': 1e-6})
        print("仓位分配：", optv['x'].round(3))
        return optv['x'].round(3)

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
            title_opts=opts.TitleOpts(title="策略收益率图(最大夏普率)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
            .render("../第二题/策略收益率(最大夏普率).html")
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
            title_opts=opts.TitleOpts(title="湾区指数和策略对比图(最大夏普率)"),
            xaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                is_scale=False,
                boundary_gap=False,
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
            .render("../第二题/湾区指数和策略对比图(最大夏普率).html")
    )


def on_data(context):

    # 获得 字长收益率的 一个
    profit = (context.account().cash['total_asset'].tolist()[0] - context.backtest_setting['initial_cash'])
    profit_rate = profit/context.backtest_setting['initial_cash']
    account_profit ={'profit_rate': 100*profit_rate, 'time': format_time(context.now)}
    context.profitArray.append(account_profit)



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
    total_asset = context.account().cash['total_asset'].tolist()[0]
    print(context.account().cash['total_asset'].tolist()[0])
    print("*"*15)
    print(context.account().position())
    print("*"*15)

    code = []
    week_ols_b = []
    week_ols_k = []
    week_ols_r = []
    week_profit_rate = []
    close = []
    time = []

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
        # print(X)
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
        close.append(target['close'].tolist())
        time.append(target['time'].tolist())

        # endFor

    week_data ={
        'time': time,
        'code': code,
        'close': close,
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
    # print(week_df)
    week_df.to_csv("../第一题/30只股票的测试{year}-{month}-{day}前一个星期的数据.csv".format(year=context.now.year, month=context.now.month, day=context.now.day))
    trade_code =[x.upper() for x in week_df['code'].tolist()[0:10]]
    print('*'*15)
    print(trade_code)
    dis = get_distribution(week_df.iloc[0:10],2)


    #  buy_code 是要买的股票 target_code 是有的股票
    target_code = []
    buy_code = []
    sold_code = []
    if context.account().cash['last_amount'][0]!=0:
        try:
            hold_code = context.account().position()['code'].tolist()
        except:
            hold_code = []
    else:
        hold_code = []

    # context.account().position()
    #  在这里将 已持仓的 继续持仓，然后卖出剩下的  然后买入选入持仓池里面的
    for i in range(0,10):
        if trade_code[i] in hold_code:
            target_code.append(trade_code[i])
            # 对索引号为0的账户中索引号为0的标的，调仓到总权益的10%，委托方式为市价委托
            # 对索引号为0的账户开仓买入10000元价值，索引号为0的标的，，委托方式为市价委托
            # order_target_value(account_idx=0, target_idx=0, value=10000.0, side=1, order_type=2, price=0.0)
            if dis[i]<1e-5:
                continue
            print("buy_old_code", trade_code[i], "     ", dis[i])
            order_target_value(account_idx=0, target_idx=context.target_list.index(trade_code[i]), target_value=dis[i]*total_asset, side=1, order_type=2, price=0.0)
            # order_target_percent(account_idx=0, target_idx=context.target_list.index(trade_code[i]), target_percent=dis[i], side=1, order_type=2, price=0.0)
        else:
            buy_code.append(trade_code[i])
        if len(hold_code)!=0  and i <len(hold_code) and not(hold_code[i] in trade_code):
            sold_code.append(hold_code[i])
            # 对索引号为0的账户中索引号为0的标的，调仓到多头0股，委托方式为市价委托
            order_target_volume(account_idx=0, target_idx=context.target_list.index(hold_code[i]), target_volume=0, side=1, order_type=2, price=0.0)
    print('hold_code:', hold_code)
    print('buy_code', buy_code)
    print('sold_code', sold_code)
    print("target_code", target_code)


    # 检测是否持仓 如果持仓 那么 平仓
    # order_close_all(account_idx=0)
    # order_cancel_all(account_indice=[0])

    #  在这里进行交易
    for i in range(0,len(buy_code)):
        # 对索引号为0的账户开仓买入10000元价值，索引号为0的标的，，委托方式为市价委托
        # every_money = context.backtest_setting['initial_cash']*0.1
        # all_money = context.account().cash['total_asset'].tolist()[0] -len(target_code) * every_money

        # print(all_money-i*every_money,'*\n',every_money)
        dis_index = trade_code.index(buy_code[i])
        # print(dis_index, dis[dis_index])
        # order_target_percent(account_idx=0, target_idx=context.target_list.index(buy_code[i]), target_percent=dis[dis_index], side=1, order_type=2, price=0.0)
            # print("+"*15)
        # print(dis[dis_index],total_asset)
        if dis[dis_index] < 1e-5:
            continue
        print("buy_new_code", buy_code[i], "     ", dis[dis_index])

        order_value(account_idx=0, target_idx=context.target_list.index(buy_code[i]), value=dis[dis_index]*total_asset, side=1, position_effect=1, order_type=2, price=0.0)
        # elif all_money-i*every_money >0:
        #     order_target_percent(account_idx=0, target_idx=context.target_list.index(buy_code[i]), target_percent=dis[dis_index], side=1, order_type=2, price=0.0)

            # print('*'*15)
            # order_percent(account_idx=0, target_idx=context.target_list.index(buy_code[i]), percent=1, side=1, position_effect=1, order_type=2, price=0.0)


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