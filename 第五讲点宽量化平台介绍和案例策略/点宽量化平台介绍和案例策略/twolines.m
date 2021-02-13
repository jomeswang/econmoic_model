function twolines(bInit,bDayBegin,cellPar)
%% 双均线策略 %%
% （1）均线为5日和20日均线
% （2）5日均线上穿20日均线，买入；
% （3）5日均线下穿20日均线，卖出。

%% 定义全局变量 %%
global get_data; 
global TLen;

%% 参数传递 %%
len = cellPar{1};      % 为滑动窗口大小
long = cellPar{2};     % 长均线参数
short = cellPar{3};    % 短均线参数
initial = cellPar{4};  % 总资金

if bInit
    %% 设置回测细节 %%
    traderSetBacktest(1000000,0,0.000025,0.02,0,1,0,0);%账号初始设置
    %% 初始化和数据注册 %%
    get_data = traderRegKData('day',1);   % 注册日数据
    TLen = length(get_data(:,1));         % 标的数量
    
else
    %% 数据提取 %%
    datas = traderGetRegKData(get_data,len+1,true);   % 获取所有标的len+1根K线的数据
    mp = traderGetAccountPositionV2(1,1:TLen);        % 持仓仓位读取
    
    for i = 1:TLen 
       %% 数据处理和逻辑计算 %%     
        data = datas(1+8*(i-1):8*i,:);            % 获取单个标的的数据
        close = data(5,:);                        % 获取收盘价       
        if isnan(close(1))                        % 无数据跳过
            return;
        end        
        LongLine = mean(close(end-long+1:end));       % 长均线
        ShortLine = mean(close(end-short+1:end));     % 短均线

        %% 仓位管理 %%
        shareNum = floor(initial/TLen/close(end)/100)*100;  % 下单交易数量，各品种平分资金
        
        %% 交易下单 %%  
        if mp(i) == 0                                     % 空仓
            if ShortLine > LongLine                       % 短均线上穿长均线，进场
                traderDirectBuyV2(1,i,shareNum,0,'market','buy1');
            end           
        elseif mp(i) > 0                                  % 持仓
            if ShortLine < LongLine                       % 短均线下穿长均线，出场
                traderPositionToV2(1,i,0,0,'market','sell1');
            end
        end
    end
end
end
