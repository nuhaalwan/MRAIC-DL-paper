% MRAIC with shallow controller

clear
n=10000;
rng("default")
u=rand(n,1);
s=3;
alpha=0.08;

Ts=0.05;
wn=2;
zeta=0.3;
p1p2=exp(-wn*zeta*2*Ts);
p1plusp2=2*(sqrt(p1p2)*cos(wn*(sqrt(1-zeta^2))*Ts));

nn=2000;
r1=ones(1,nn/4); r2=0.5*r1;r3=r1; r4=r2;
comm=[r1 r2 r3 r4];
% First-order reference model:
%commref=0.05*filter(1,[1 -0.95],comm);
% Second-order reference model:
commref=0.0025*filter([1 2 1],[1 -p1plusp2 p1p2],comm);

rng('default')

W1=2*rand(5,s)-1;
W2=2*rand(5,5)-1;
W3=2*rand(5,5)-1;
W4=2*rand(5,5)-1;
W5=2*rand(1,5)-1;

WW1=2*rand(5,s)-1;
WW2=2*rand(5,5)-1;
WW3=2*rand(5,5)-1;
WW4=2*rand(5,5)-1;
WW5=2*rand(1,5)-1;

WWW1=2*rand(5,s)-1;
% WWW2=2*rand(5,5)-1;
% WWW3=2*rand(5,5)-1;
% WWW4=2*rand(5,5)-1;
% WWW5=2*rand(1,5)-1;
WWW2=2*rand(1,5)-1;


for k=1:n

        if k==1
            c(k)=u(k);
            %c(k)=0;
        end
        if k==2
            c(k)=u(k);
            %c(k)=u(k-1);
        end
        if k>=3 
%             
        c(k)=u(k)+c(k-1)*c(k-2)/(1+(c(k-1))^3);
        end
% %     end
    
    for hh=1:s
        c1(hh)=0;
    end
    
    for k11=1:min([s,k])
        c1(k11)=u(k-k11+1);
    end
    
    vv1=W1*c1';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=W2*yy1;
    yy2=ReLU(vv2);
    %yy2=sigmoid(vv2);
    
    vv3=W3*yy2;
    yy3=ReLU(vv3);
    %yy3=sigmoid(vv3);
    
    vv4=W4*yy3;
    yy4=ReLU(vv4);
    %yy4=sigmoid(vv4);
    
    vvv=W5*yy4;
    v1(k)=vvv;

    d(k)=c(k);

    
    e(k)=d(k)-vvv;
    delta=e(k);
    
    e4= W5'*delta;
    delta4=(vv4>0).*e4;
    %delta4=y4.*(1-y4).*e4;
    
    e3=W4'*delta4;
    delta3=(vv3>0).*e3;
    %delta3=y3.*(1-y3).*e3;
    
    e2=W3'*delta3;
    delta2=(vv2>0).*e2;
    %delta2=y2.*(1-y2).*e2;
    
    e1=W2'*delta2;
    delta1=(vv1>0).*e1;
    %delta1=y1.*(1-y1).*e1;
    
    dW5=alpha*delta*yy4';
    W5=W5+dW5;
    
    dW4=alpha*delta4*yy3';
    W4=W4+dW4;
    
    dW3=alpha*delta3*yy2';
    W3=W3+dW3;
    
    dW2=alpha*delta2*yy1';
    W2=W2+dW2;
    
    dW1=alpha*delta1*c1;
    W1=W1+dW1;

    for hh=1:s
        c2(hh)=0;
    end
    
    for k11=1:min([s,k])
        c2(k11)=v1(k-k11+1);
    end
    
    vv1=WW1*c2';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=WW2*yy1;
    yy2=ReLU(vv2);
    %yy2=sigmoid(vv2);
    
    vv3=WW3*yy2;
    yy3=ReLU(vv3);
    %yy3=sigmoid(vv3);
    
    vv4=WW4*yy3;
    yy4=ReLU(vv4);
    %yy4=sigmoid(vv4);
    
    vvv=WW5*yy4;


    d(k)=u(k);

    
    ee(k)=d(k)-vvv;
    delta=ee(k);
    
    e4= WW5'*delta;
    delta4=(vv4>0).*e4;
    %delta4=y4.*(1-y4).*e4;
    
    e3=WW4'*delta4;
    delta3=(vv3>0).*e3;
    %delta3=y3.*(1-y3).*e3;
    
    e2=WW3'*delta3;
    delta2=(vv2>0).*e2;
    %delta2=y2.*(1-y2).*e2;
    
    e1=WW2'*delta2;
    delta1=(vv1>0).*e1;
    %delta1=y1.*(1-y1).*e1;
    
    dWW5=alpha*delta*yy4';
    WW5=WW5+dWW5;
    
    dWW4=alpha*delta4*yy3';
    WW4=WW4+dWW4;
    
    dWW3=alpha*delta3*yy2';
    WW3=WW3+dWW3;
    
    dWW2=alpha*delta2*yy1';
    WW2=WW2+dWW2;
    
    dWW1=alpha*delta1*c1;
    WW1=WW1+dWW1;
    
end

for k=1:n
    es(k)=e(k)^2;
    sum=0;
    for i=1:k
        sum=sum+es(i);
    end
    zi(k)=sum/k;
end

for k=1:n
    ees(k)=ee(k)^2;
    sum=0;
    for i=1:k
        sum=sum+ees(i);
    end
    zzi(k)=sum/k;
end
 
% t=1:n;
% figure(1)
% plot(t,zi,'k','linewidth',1.5)
% axis([0 10000 0 2])
% 
% figure(2)
% plot(t,zzi,'r','linewidth',1.5)
% axis([0 10000 0 2])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Online: Figure 4

clear c
clear u
clear v1

for k=1:nn

% C^ copy

for hh=1:s
        c2(hh)=0;
    end
    
    for k11=1:min([s,k])
        c2(k11)=comm(k-k11+1);
    end
    
    vv1=WWW1*c2';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=WWW2*yy1;

u(k)=vv2;

% Plant:

 if k>nn*0.6
    c(k)=vv2+c(k-1)*c(k-2)/(1+4*(c(k-1))^3);
     else
        if k==1
            c(k)=vv2;
            %c(k)=0;
        end
        if k==2
            c(k)=vv2;
            %c(k)=u(k-1);
        end
       if k>=3 && k<=nn*0.6
     %   if k>=3              
        c(k)=vv2+c(k-1)*c(k-2)/(1+(c(k-1))^3);
        end
     end

%%%%%%%%%%%%%%%%%%%%%%
% Iden. and inv. iden.
%%%%%%%%%%%%%%%%%%%%%%

for hh=1:s
        c1(hh)=0;
    end
    
    for k11=1:min([s,k])
        c1(k11)=u(k-k11+1);
    end
    
    vv1=W1*c1';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=W2*yy1;
    yy2=ReLU(vv2);
    %yy2=sigmoid(vv2);
    
    vv3=W3*yy2;
    yy3=ReLU(vv3);
    %yy3=sigmoid(vv3);
    
    vv4=W4*yy3;
    yy4=ReLU(vv4);
    %yy4=sigmoid(vv4);
    
    vvv=W5*yy4;
    v1(k)=vvv;

    d(k)=c(k);

    
    e(k)=d(k)-vvv;
    delta=e(k);
    
    e4= W5'*delta;
    delta4=(vv4>0).*e4;
    %delta4=y4.*(1-y4).*e4;
    
    e3=W4'*delta4;
    delta3=(vv3>0).*e3;
    %delta3=y3.*(1-y3).*e3;
    
    e2=W3'*delta3;
    delta2=(vv2>0).*e2;
    %delta2=y2.*(1-y2).*e2;
    
    e1=W2'*delta2;
    delta1=(vv1>0).*e1;
    %delta1=y1.*(1-y1).*e1;
    
    dW5=alpha*delta*yy4';
    W5=W5+dW5;
    
    dW4=alpha*delta4*yy3';
    W4=W4+dW4;
    
    dW3=alpha*delta3*yy2';
    W3=W3+dW3;
    
    dW2=alpha*delta2*yy1';
    W2=W2+dW2;
    
    dW1=alpha*delta1*c1;
    W1=W1+dW1;

    for hh=1:s
        c2(hh)=0;
    end
    
    for k11=1:min([s,k])
        c2(k11)=v1(k-k11+1);
    end
    
    vv1=WW1*c2';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=WW2*yy1;
    yy2=ReLU(vv2);
    %yy2=sigmoid(vv2);
    
    vv3=WW3*yy2;
    yy3=ReLU(vv3);
    %yy3=sigmoid(vv3);
    
    vv4=WW4*yy3;
    yy4=ReLU(vv4);
    %yy4=sigmoid(vv4);
    
    vvv=WW5*yy4;


    d(k)=u(k);

    
    ee(k)=d(k)-vvv;
    delta=ee(k);
    
    e4= WW5'*delta;
    delta4=(vv4>0).*e4;
    %delta4=y4.*(1-y4).*e4;
    
    e3=WW4'*delta4;
    delta3=(vv3>0).*e3;
    %delta3=y3.*(1-y3).*e3;
    
    e2=WW3'*delta3;
    delta2=(vv2>0).*e2;
    %delta2=y2.*(1-y2).*e2;
    
    e1=WW2'*delta2;
    delta1=(vv1>0).*e1;
    %delta1=y1.*(1-y1).*e1;
    
    dWW5=alpha*delta*yy4';
    WW5=WW5+dWW5;
    
    dWW4=alpha*delta4*yy3';
    WW4=WW4+dWW4;
    
    dWW3=alpha*delta3*yy2';
    WW3=WW3+dWW3;
    
    dWW2=alpha*delta2*yy1';
    WW2=WW2+dWW2;
    
    dWW1=alpha*delta1*c1;
    WW1=WW1+dWW1;
    

%%%%%%%%%%%%%%%%%%%



% P^-1

for hh=1:s
        c2(hh)=0;
    end
    
    for k11=1:min([s,k])
        c2(k11)=c(k-k11+1);
    end
    
    vv1=WW1*c2';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=WW2*yy1;
    yy2=ReLU(vv2);
    %yy2=sigmoid(vv2);
    
    vv3=WW3*yy2;
    yy3=ReLU(vv3);
    %yy3=sigmoid(vv3);
    
    vv4=WW4*yy3;
    yy4=ReLU(vv4);
    %yy4=sigmoid(vv4);
    
    vvv2=WW5*yy4;

% Other P^-1

for hh=1:s
        c2(hh)=0;
    end
    
    for k11=1:min([s,k])
        c2(k11)=commref(k-k11+1);
    end
    
    vv1=WW1*c2';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=WW2*yy1;
    yy2=ReLU(vv2);
    %yy2=sigmoid(vv2);
    
    vv3=WW3*yy2;
    yy3=ReLU(vv3);
    %yy3=sigmoid(vv3);
    
    vv4=WW4*yy3;
    yy4=ReLU(vv4);
    %yy4=sigmoid(vv4);
    
    vvv1=WW5*yy4;


% C^

for hh=1:s
        c2(hh)=0;
    end
    
    for k11=1:min([s,k])
        c2(k11)=comm(k-k11+1);
    end
    
    vv1=WWW1*c2';
    yy1=ReLU(vv1);
    %yy1=sigmoid(vv1);
    
    vv2=WWW2*yy1;


   % d(k)=u(k);

    
    ef(k)=vvv1-vvv2;
    delta=ef(k);
    
    
    e1=WWW2'*delta;
    delta1=(vv1>0).*e1;
    %delta1=y1.*(1-y1).*e1;
    
    dWWW2=alpha*delta*yy1';
    WWW2=WWW2+dWWW2;
    
    dWWW1=alpha*delta1*c1;
    WWW1=WWW1+dWWW1;
    
end

t=1:nn;
figure(3)
plot(t,c,'r',t,commref,'k--','linewidth',1.5)
%plot(t,u,'k','linewidth',1.5)
axis([0 nn -0.5 2])
%axis([0 nn -0.5 1])
xlabel('time instant','FontWeight','bold')
ylabel('model reference and plant output signals','FontWeight','bold')
%ylabel('control signal','FontWeight','bold')
legend('plant output','model reference output','FontWeight','bold')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=ReLU(x)
y = max(0,x);
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=sigmoid(x)
y=1./(1+exp(-x));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ym=Dropout(y,rat)
[m,n]=size(y);
ym=zeros(m,n);
num=round(m*n*(1-rat));
idx=randperm(m*n,num);
ym(idx)=1/(1-rat);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

