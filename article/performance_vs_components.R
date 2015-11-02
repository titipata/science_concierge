library(ggplot2)
library(reshape2)
library(ggthemes)
library(scales)
library(grid)
library(plyr)

mean_point <- stat_summary(fun.y = 'mean', geom = 'point', position = position_dodge(width = 0.1))
mean_line <- stat_summary(fun.y = 'mean', geom = 'line', position = position_dodge(width = 0.1))
se_error <- stat_summary(fun.data = 'mean_se', geom = 'errorbar', width = 0.05, position = position_dodge(width = 0.1))

experiments <- melt(read.csv('data//components_vs_distance.csv'),
                    id.vars = c('poster_number', 'distance', 'n_components'))

pdf('figures/performance_vs_components.pdf', width=5, height=3)
ggplot(experiments, aes(x = n_components, y = distance))  + 
  mean_point + 
  mean_line + 
  se_error + 
  scale_color_brewer(palette = 6, type = 'qual', name = 'Algorithm') +
  ylab('Topic distance') +
  xlab('Number of components') +
  theme_classic() +
  coord_cartesian(xlim=c(0, 550))

dev.off()
