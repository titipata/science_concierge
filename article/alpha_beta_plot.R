pdf('alpha_beta_relation_plot.pdf', width=6, height=3)

jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

ggplot(alpha_beta, aes(x = alpha, y = beta, fill=avg_distance)) + 
  geom_tile(interpolate = T) + 
  scale_fill_gradientn(colours = jet.colors(10)) + 
  facet_wrap(~distance_away) + 
  theme_minimal()

dev.off()
