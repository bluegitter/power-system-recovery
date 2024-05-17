clc
clear
%可以正负变化
%14节点数据
% name Pload Qload
bus=[1	0	0;
     2	21.7	12.7;
     3	94.2	19;
     4	47.8	-3.9;
     5	7.6	1.6;
     6	11.2	7.5;
     7	0	0;
     8	0	0;
     9	29.5	16.6;
     10	3.5	5.8;
     11	3.5	1.8;
     12	6.1	1.6;
     13	13.5	5.8;
     14	14.9	5];
%Gbus   Pmax    Ps  r    t2_1  t3  Pmin   Th
 gen=[1	332.4	0	1.11 0     0    0     900;
      2	140     5	1.37 6     11   34.25 900;
      3	100     4	1.27 10    15   31.75 900;
      % 6	100     3	1.17 8     13   29.25 900;
      % 8	100     2.3 0.87 4     10   26.1  900
      ];

 genwi6=[0.27 0.27 0.27 0.38 0.38 0.38 0.32 0.32 0.32 0.49 0.49 0.49 0.49 0.49 0.49 0.44 0.44 0.44 0.44 0.44 0.44 0.41 0.41 0.41];
 genpv8=[0.96 0.96 0.96 1.85 1.85 1.85 2.44 2.44 2.44 3.41 3.41 3.41 3.74 3.74 3.74 4.18 4.18 4.18 4.11 4.11 4.11 5.40 5.40 5.40];
 
%branch node1 node2
bra=[1	1	2;
     2	1	5;
     3	2	3;
     4	2	4;
     5	2	5;
     6	3	4;
     7	4	5;
     8	4	7;
     9	4	9;
    10	5	6;
    11	6	11;
    12	6	12;
    13	6	13;
    14	7	8;
    15	7	9;
    16	9	10;
    17	9	14;
    18	10	11;
    19	12	13;
    20	13	14
];


T=120;
dt=5;
Nt=T/dt;

Nbus=size(bus,1);
Nbra=size(bra,1);
Ngen=size(gen,1);

M=9999;

IM = zeros(Nbra,Nbus);
for l = 1:Nbra
    IM(l,bra(l,2)) = 1;
    IM(l,bra(l,3)) = 1;
end

t=ones(Nt*Ngen,1)*dt;
tl=ones(Nt,1)*dt;


cvx_begin sdp
cvx_solver mosek

variable P(Nt*Ngen)      
variable a(Nbus,Nt) binary
variable b(Nbra,Nt) binary                                                  
variable k(Ngen,Nt) binary                                                  %是否启动
variable x(Ngen,Nt) binary                                                  %是否超过t3
variable y(Ngen,Nt) binary                                                  %是否超过t2
variable PL(Nbus,Nt) 
variable kk(2,Nt)   binary

maximize (P'*t + genwi6*kk(1,:)'*dt + genpv8*kk(2,:)'*dt)
% maximize (PL(11,:)*tl)
% maximize (P'*t + 50*PL(10,:)*tl + 50*PL(11,:)*tl)
% maximize (P'*t + 63*PL(10,:)*tl) % + 63*PL(11,:)*tl)
subject to
%连接性和次序约束

for tt = 1:Nt-1
    g=0;
    gg=0;
    for i = 1:Nbus 
        if i == 1 || i == 2 || i == 3
            g=g+1;
            k(g,tt) <= a(i,tt);  %k(g,tt+1) <= a(i,tt);
        elseif i == 6 || i == 8
            gg=gg+1;
            kk(gg,tt) <= a(i,tt);
        end
        a(i,tt+1) <= b(:,tt+1)'*IM(:,i);
    end
    for l = 1:Nbra
        b(l,tt+1) <= a(bra(l,2),tt) + a(bra(l,3),tt);
    end
end


for tt = 1:Nt-1 
    for i = 1:Nbus
        a(i,tt) <= a(i,tt+1);
    end
    for g = 1:Ngen
        k(g,tt) <= k(g,tt+1);
        x(g,tt) <= x(g,tt+1);
        y(g,tt) <= y(g,tt+1);
    end
    for gg = 1:2
        kk(gg,tt) <= kk(gg,tt+1);
    end
    for l = 1:Nbra
        b(l,tt) <= b(l,tt+1);      
    end
end


for i = 1:Ngen
    if i == 1
        k(i,1) == 1;
    else
        k(i,1) == 0;
    end
end
for i = 2:Nbus
    a(i,1) == 0;
end
for l = 1:Nbra
    b(l,1) == 0;
end
for gg = 1:2
    kk(gg,1) == 0;
end


for tt = 1:Nt-1
    scha = 0;
    for l = 1:Nbra
        cha = b(l,tt+1)-b(l,tt);
        scha = scha + cha;
    end
    scha <= 1;
end

for i = 1:Ngen
    for tt = 1:Nt
        (sum(k(i,1:tt))-gen(i,5))/T <= y(i,tt);
        y(i,tt) <= (sum(k(i,1:tt))-gen(i,5)-1)/T+1;  
        (sum(k(i,1:tt))-gen(i,6))/T <= x(i,tt);
        x(i,tt) <= (sum(k(i,1:tt))-gen(i,6)-1)/T+1;
    end
end


for i = 1:Ngen
    for tt = 1:Nt

        -k(i,tt)*M <= P(i+Ngen*(tt-1)); 
        P(i+Ngen*(tt-1)) <= M*k(i,tt); 
               
        P(i+Ngen*(tt-1)) >=  gen(i,4)*(sum(y(i,1:tt)))*dt-k(i,tt)*gen(i,3)-x(i,tt)*M;
        P(i+Ngen*(tt-1)) <=  gen(i,4)*(sum(y(i,1:tt)))*dt-k(i,tt)*gen(i,3)+x(i,tt)*M;
        
        -(1-x(i,tt))*M+gen(i,7)-k(i,tt)*gen(i,3) <= P(i+Ngen*(tt-1));
        P(i+Ngen*(tt-1)) <= (1-x(i,tt))*M+gen(i,2)-k(i,tt)*gen(i,3);
        
        sum(tt-k(i,1:tt))+1 <= gen(i,8);                                       %机组热启动时限约束
    end
end
for i = 1:Ngen
    for tt = 1:Nt-1
        -dt*gen(i,4) <= P(i+Ngen*tt)-P(i+Ngen*(tt-1));                             %机组爬坡率约束
        P(i+Ngen*tt)-P(i+Ngen*(tt-1)) <= dt*gen(i,4);
    end
end




% %非黑启动机组启动功率约束
% for tt = 1:Nt
%     sum(P(1+Ngen*(tt-1):Ngen*tt)) - PL(10,tt) - PL(11,tt) >= 0;
% end
%系统功率平衡约束
for tt = 1:Nt
    for i = 1:Nbus
        if i == 11 %|| i == 11
%             -a(i,tt)*bus(i,2) <= PL(i,tt);
            -3.5*a(i,tt) <= PL(i,tt);
            PL(i,tt) <= a(i,tt)*bus(i,2);
        else
            0 <= PL(i,tt);
            PL(i,tt) <= a(i,tt)*bus(i,2);
        end
    end
    sum(P(1+Ngen*(tt-1):Ngen*tt)) + genwi6(tt)*kk(1,tt) + genpv8(tt)*kk(2,tt) - sum(PL(:,tt)) == 0;
end
%已恢复的负荷不再切除
for tt = 1:Nt-1
    for i = 1:Nbus
        if i ~= 11 %|| i ~= 11
            PL(i,tt) <= PL(i,tt+1);
        end
    end
end

cvx_end

P'*t + genwi6*kk(1,:)'*dt + genpv8*kk(2,:)'*dt
