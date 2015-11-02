library(ggplot2)

tsne_data <- melt(read.csv('data//tsne_2d.csv'),
                  id.vars = c('x_tsne', 'y_tsne', 'topic'))

pdf('figures/component_visualization.pdf', width=5, height=3)
ggplot(tsne_data, aes(x = x_tsne, y = y_tsne, color=topic)) + 
  geom_point(alpha=0.5, size=1) + 
  scale_color_brewer(palette = 6, type = 'qual', name = 'Topic') +
  theme_classic() +
  ylab('Component 2') +
  xlab('Component 1')

dev.off()