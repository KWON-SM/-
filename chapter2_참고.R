#################################################################### 
#  참고자료  
#

sqrt(0.85*(1-0.85))/sqrt(1)
#[1] 0.3570714
sqrt(0.85*(1-0.85))/sqrt(4)
#[1] 0.1785357
sqrt(0.85*(1-0.85))/sqrt(16)
#[1] 0.08926786
sqrt(0.85*(1-0.85))/sqrt(25)
#[1] 0.07141428
sqrt(0.85*(1-0.85))/sqrt(100)
#[1] 0.03570714
sqrt(0.85*(1-0.85))/sqrt(10000)
#[1] 0.003570714


pnorm(1, mean = 0, sd = 1, lower.tail = TRUE)    # 정규분포의 누적분포함수
#[1] 0.8413447
1 - pnorm(1, mean = 0, sd = 1, lower.tail = TRUE)
#[1] 0.1586553


library(ggplot2)
ggplot(data.frame(x=c(-4,4)), aes(x=x)) +
  stat_function(fun=dnorm, colour="blue", size=1) +
  stat_function(fun=dt, args=list(df=3), colour="red", size=1) +
  stat_function(fun=dt, args=list(df=1), colour="black", size=1) +
  annotate("segment", x=1.5, xend=2, y=0.4, yend=0.4, colour="blue", size=1) +
  annotate("segment", x=1.5, xend=2, y=0.37, yend=0.37, colour="red", size=1) + 
  annotate("segment", x=1.5, xend=2, y=0.34, yend=0.34, colour="black", size=1) + 
  annotate("text", x=2.4, y=0.4, label="N(0,1)") +
  annotate("text", x=2.4, y=0.37, label="t(,3)") + 
  annotate("text", x=2.4, y=0.34, label="t(,1)") + 
  ggtitle("Normal Distribution, t-distribution")



# t분포 분위수 함수 : qt(p, df, lower.tail = TRUE/FALSE) 
# Pr(t > 0.025)이고, 자유도가 4 
# lower.tail  logical; if TRUE (default), 
#                         probabilities are P[X ≤ x] otherwise, P[X > x].
#
(q <- qt(p=0.025, df=4, lower.tail = FALSE))
# [1] 2.776445

pt(q=q, df=4, lower.tail = FALSE)                    
# [1] 0.025



factorial(4) / (factorial(2) * factorial(2))
#[1] 6
choose(4, 2)
#[1] 6



choose(3, 0)/2^3     # 동전 앞면이 나오는 횟수 0 
#[1] 0.125
choose(3, 1)/2^3     # 동전 앞면이 나오는 횟수 1 
#[1] 0.375
choose(3, 2)/2^3     # 동전 앞면이 나오는 횟수 2 
#[1] 0.375
choose(3, 3)/2^3     # 동전 앞면이 나오는 횟수 3 
#[1] 0.125

dbinom(0:3, 3, 0.5)                  #  이항분포의 확률밀도함수 이용 
#[1] 0.125 0.375 0.375 0.125



(y <- dbinom(0:20, size=20, prob=0.5))
# [1] 9.536743e-07 1.907349e-05 1.811981e-04 1.087189e-03 4.620552e-03
# [6] 1.478577e-02 3.696442e-02 7.392883e-02 1.201344e-01 1.601791e-01
#[11] 1.761971e-01 1.601791e-01 1.201344e-01 7.392883e-02 3.696442e-02
#[16] 1.478577e-02 4.620552e-03 1.087189e-03 1.811981e-04 1.907349e-05
#[21] 9.536743e-07
plot(0:20, y, type='h', lwd=5, col="grey", ylab="Probability", xlab="확률변수 X", 
     main = c("X ~ B(20, 0.5)"))



#동전 앞면이 나올 확률이 0.4인 동전을 10회 던져서 앞이 3회 나올 확률
choose(10, 3) * 0.4^3 * 0.6^7
#[1] 0.2149908

dbinom(3, 10, 0.4)
#[1] 0.2149908



pbinom(12, size=20, prob=0.5, lower.tail = FALSE)
#[1] 0.131588

1 - pbinom(12, size=20, prob=0.5, lower.tail = TRUE)  
#[1] 0.131588



pbinom(12, size=20, prob=0.5, lower.tail = TRUE)    # 이항분포의 누적분포함수
#[1] 0.868412 

sum(dbinom(0:12, size=20, prob=0.5))                    # 이항분포의 확률밀도함수
#[1] 0.868412 



dbinom(0:10, 10, 0.5)
#[1] 0.0009765625 0.0097656250 0.0439453125 0.1171875000 0.2050781250 0.2460937500 0.2050781250 0.1171875000 0.0439453125 0.0097656250
#[11] 0.0009765625

barplot(dbinom(0:10, 10, 0.5), names.arg = 0:10, las=1, ylim = c(0.0, 0.25))



(p <- pbinom(3, 6, 0.5))       # 이항분포의 누적분포함수 
#[1] 0.65625

(r <- qbinom(p, 6, 0.5))        # 이항분포의 분위수함수
#[1] 3



