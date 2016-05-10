library(ggplot2)
library(reshape2)
library(ggthemes)
library(scales)
library(grid)
library(plyr)

mean_point <- stat_summary(fun.y = 'mean', geom = 'point', position = position_dodge(width = 0.0))
mean_line <- stat_summary(fun.y = 'mean', geom = 'line', position = position_dodge(width = 0.0))
se_error <- stat_summary(fun.data = 'mean_se', geom = 'errorbar', width = 0.05, position = position_dodge(width = 0.0))

experiments <- read.csv('data/human_vs_topic_distance.csv')
experiments$tfidf_distance <- scale(experiments$tfidf_distance)
experiments$keyword_distance <- scale(experiments$keyword_distance)
experiments$wordvec_distance <- scale(experiments$wordvec_distance)
experiments$countvec_distance <- scale(experiments$countvec_distance)
experiments$logentropy_distance <- scale(experiments$logentropy_distance)
experiments_melt <- melt(experiments, id.vars = c('poster_number', 'human_distance'))

experiments_melt$variable <- revalue(experiments_melt$variable,
                                    c("tfidf_distance"="tf-idf",
                                      "keyword_distance"="keyword",
                                      "countvec_distance"="term frequency",
                                      "logentropy_distance"="log-entropy",
                                      "wordvec_distance"="average word vector"))

experiments_melt$variable <- factor(experiments_melt$variable,
                                   c('term frequency', 'tf-idf', 'log-entropy',
                                     'average word vector', 'keyword'))

pdf('figures/human_vs_topic_distance.pdf', width=6, height=3)
ggplot(experiments_melt, aes(x = human_distance, y = value, color=variable))  +
  mean_point +
  mean_line +
  se_error +
  scale_color_brewer(palette = 6, type = 'qual', name = 'Algorithm') +
  ylab('Topic distance (z-score)') +
  xlab('Human curated topic distance') +
  theme_classic()
dev.off()
pdf('figures/human_vs_topic_distance.pdf', width=6, height=3)

# Spearman correlation
cor.test(experiments$human_distance, experiments$abstact_distance, method = "spearman")
cor.test(experiments$human_distance, experiments$keyword_distance, method = "spearman")
