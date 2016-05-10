library(plyr)
library(contrast)
library(reshape2)

# load the experiment
experiments  <- read.csv('data/multiple_votes.csv')

# keyword is better than random
t.test(experiments$avg_random, experiments$avg_node_distance_kw, paired = T)

# keyword improves with votes
summary(lm(avg_node_distance_kw ~ number_recommend, experiments))

# science concierge is better than keyword
t.test(experiments$avg_node_distance_kw, experiments$avg_node_distance , paired = T)

# term frequency
t.test(experiments$avg_node_distance_cv, experiments$avg_node_distance , paired = T)

# log entropy
t.test(experiments$avg_node_distance_le, experiments$avg_node_distance , paired = T)

# word vector
t.test(experiments$avg_node_distance_wv, experiments$avg_node_distance , paired = T)


# scholarfy improves with votes
summary(lm(avg_node_distance ~ number_recommend, experiments))


# melt experiment data
melted_experiment <- melt(experiments, 
                          id.vars = c('number_recommend'), 
                          measure.vars = c('avg_node_distance', 'avg_node_distance_kw'), 
                          variable.name = 'method', 
                          value.name = 'distance')

melted_experiment$method <- factor(melted_experiment$method)

# keyword reduces error significantly less than scholarfy
summary(lm(distance ~ 1 + number_recommend + method:number_recommend, melted_experiment))
