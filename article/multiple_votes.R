library(ggplot2)
library(reshape2)
library(ggthemes)
library(scales)
library(grid)
library(plyr)

plot_theme <- theme_classic() +
  theme(text = element_text(size = 20), 
        axis.title.x = element_text(vjust = -1, face = 'bold'), 
        axis.title.y = element_text(vjust = 2, face = 'bold'), 
        plot.margin =  (unit(c(2, 2, 2, 2), "cm")),
        legend.position = c(1, 0), legend.justification = c(1, 0))


mean_point <- stat_summary(fun.y = 'mean', geom = 'point', position = position_dodge(width = 0.1))
mean_line <- stat_summary(fun.y = 'mean', geom = 'line', position = position_dodge(width = 0.1))
se_error <- stat_summary(fun.data = 'mean_se', geom = 'errorbar', width = 0.05, position = position_dodge(width = 0.1))

experiments <- melt(read.csv('data//poster_node_distance.csv'),
                   id.vars = c('poster_number', 'number_recommend'))

experiments$variable <- revalue(experiments$variable, 
                                c("avg_node_distance"="Science Concierge",
                                "avg_node_distance_kw"="Keyword",
                                "avg_random"="Random"))

experiments$variable <- factor(experiments$variable, 
                               c('Random', 'Keyword', 'Science Concierge'))

pdf('figures/performance_vs_votes.pdf', width=6, height=3)
ggplot(experiments, 
       aes(x = number_recommend, y = value, color = variable)) + 
  mean_point + 
  mean_line + 
  se_error + 
  theme_classic() +
  scale_y_continuous(breaks = c(1, 2, 3)) + 
  coord_cartesian(ylim = c(0, 3)) +
  scale_color_brewer(palette = 6, type = 'qual', name = 'Algorithm') + 
  ylab('Topic distance') +
  xlab('Number of votes')  
dev.off()
