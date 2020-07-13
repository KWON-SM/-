####### chapter 2 
PSDS_PATH <- file.path('./source')

### chapter2.3
loans_income <- read.csv(file.path(PSDS_PATH, 'data', 'loans_income.csv'))[,1]


# 단순랜덤표본
samp_data <- data.frame(income=sample(loans_income, 1000), 
                        type='data_dist')

# 5개 표본씩 평균
samp_mean_05 <- data.frame(
  income = tapply(sample(loans_income, 1000*5), 
                  rep(1:1000, rep(5, 1000)), FUN=mean),
  type = 'mean_of_5')

# 20개 표본씩 평균
samp_mean_20 <- data.frame(
  income = tapply(sample(loans_income, 1000*20), 
                  rep(1:1000, rep(20, 1000)), FUN=mean),
  type = 'mean_of_20')

# bind the data.frames and convert type to a factor
income <- rbind(samp_data, samp_mean_05, samp_mean_20)
income$type = factor(income$type, 
                     levels=c('data_dist', 'mean_of_5', 'mean_of_20'),
                     labels=c('Data', 'Mean of 5', 'Mean of 20'))

# plot the histograms
library(ggplot2)
ggplot(income, aes(x=income)) +
  geom_histogram(bins=40) +
  facet_grid(type ~ .)



### chapter2.4

library(boot)
PSDS_PATH <- file.path('./source')
loans_income <- read.csv(file.path(PSDS_PATH, 'data', 'loans_income.csv'))[,1]
 
stat_fun <- function(x, idx) median(x[idx])
boot_obj <- boot(loans_income, R = 1000, statistic=stat_fun) # 부트스트랩

boot_obj

##########################################

