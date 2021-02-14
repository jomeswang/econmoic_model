function twolines(bInit,bDayBegin,cellPar)
%% ˫���߲��� %%
% ��1������Ϊ5�պ�20�վ���
% ��2��5�վ����ϴ�20�վ��ߣ����룻
% ��3��5�վ����´�20�վ��ߣ�������

%% ����ȫ�ֱ��� %%
global get_data; 
global TLen;

%% �������� %%
len = cellPar{1};      % Ϊ�������ڴ�С
long = cellPar{2};     % �����߲���
short = cellPar{3};    % �̾��߲���
initial = cellPar{4};  % ���ʽ�

if bInit
    %% ���ûز�ϸ�� %%
    traderSetBacktest(1000000,0,0.000025,0.02,0,1,0,0);%�˺ų�ʼ����
    %% ��ʼ��������ע�� %%
    get_data = traderRegKData('day',1);   % ע��������
    TLen = length(get_data(:,1));         % �������
    
else
    %% ������ȡ %%
    datas = traderGetRegKData(get_data,len+1,true);   % ��ȡ���б��len+1��K�ߵ�����
    mp = traderGetAccountPositionV2(1,1:TLen);        % �ֲֲ�λ��ȡ
    
    for i = 1:TLen 
       %% ���ݴ�����߼����� %%     
        data = datas(1+8*(i-1):8*i,:);            % ��ȡ������ĵ�����
        close = data(5,:);                        % ��ȡ���̼�       
        if isnan(close(1))                        % ����������
            return;
        end        
        LongLine = mean(close(end-long+1:end));       % ������
        ShortLine = mean(close(end-short+1:end));     % �̾���

        %% ��λ���� %%
        shareNum = floor(initial/TLen/close(end)/100)*100;  % �µ�������������Ʒ��ƽ���ʽ�
        
        %% �����µ� %%  
        if mp(i) == 0                                     % �ղ�
            if ShortLine > LongLine                       % �̾����ϴ������ߣ�����
                traderDirectBuyV2(1,i,shareNum,0,'market','buy1');
            end           
        elseif mp(i) > 0                                  % �ֲ�
            if ShortLine < LongLine                       % �̾����´������ߣ�����
                traderPositionToV2(1,i,0,0,'market','sell1');
            end
        end
    end
end
end
