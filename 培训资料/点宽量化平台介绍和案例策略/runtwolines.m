clear;
clc;
%% 标的设置 %%
targetList(1).Market = 'sse';
targetList(1).Code = '600000'; % 浦发银行
targetList(2).Market = 'sse';
targetList(2).Code = '601699'; % 潞安环能

%% 参数设置 %%
len = 20;                % 为滑动窗口大小
long = 20;               % 长均线参数，20日均线
short = 5;               % 短均线参数，5日均线
Freq = 1;                % 刷新频率，每天刷新
begintime = 20190101;    % 开始回测时间
endtime = 20190701;      % 结束回测时间

%% 交易账户定义 %%
AccountList(1) = {'StockBackReplay'};    % 回测账户
initial = 100000000;                     % 账户资金

%% 回测函数 %%
traderRunBacktestV2('twolines',@twolines,{len,long,short,initial},AccountList,targetList,'day',Freq,begintime,endtime,'FWard'); 