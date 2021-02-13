clear;
clc;
%% ������� %%
targetList(1).Market = 'sse';
targetList(1).Code = '600000'; % �ַ�����
targetList(2).Market = 'sse';
targetList(2).Code = '601699'; % º������

%% �������� %%
len = 20;                % Ϊ�������ڴ�С
long = 20;               % �����߲�����20�վ���
short = 5;               % �̾��߲�����5�վ���
Freq = 1;                % ˢ��Ƶ�ʣ�ÿ��ˢ��
begintime = 20190101;    % ��ʼ�ز�ʱ��
endtime = 20190701;      % �����ز�ʱ��

%% �����˻����� %%
AccountList(1) = {'StockBackReplay'};    % �ز��˻�
initial = 100000000;                     % �˻��ʽ�

%% �ز⺯�� %%
traderRunBacktestV2('twolines',@twolines,{len,long,short,initial},AccountList,targetList,'day',Freq,begintime,endtime,'FWard'); 