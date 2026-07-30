library(R.utils)
library(logger)
library(optparse)

library(khroma)

vibrant <- color("vibrant")
palette(vibrant(7))

source("src/custom/common.R")

log_info("XAI Shapley Cluster - Bikeshare Dataset")

# Init
mmethod <- "rf" # Which regression model to use
log_info("method: {mmethod}")

option_list <- list(
  make_option(
    c("--prediction-accuracy"),
    type = "logical",
    default = TRUE,
    help = "Use prediction accuracy [default %default]"
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

prediction_accuracy <- opt$`prediction-accuracy`
log_info("prediction_accuracy: {prediction_accuracy}")

# Load Bikeshare dataset
data("Bikeshare", package = "ISLR2")

# Remove incomplete cases
Bikeshare <- Bikeshare[complete.cases(Bikeshare), ]
# write.csv(Bikeshare, "datasets/Bikeshare.csv", row.names = TRUE)

# Clusters by month
mnth_labels <- sort(unique(Bikeshare$mnth))
K <- length(mnth_labels)
log_info("Number of clusters: {K}")

# Build data matrix: y, x1..x8 (all predictors), xS (mnth)
mdata_full <- cbind(
  y = Bikeshare$bikers,
  x1 = Bikeshare$hr,
  x2 = Bikeshare$holiday,
  x3 = Bikeshare$weekday,
  x4 = Bikeshare$workingday,
  x5 = Bikeshare$weathersit,
  x6 = Bikeshare$temp,
  x7 = Bikeshare$hum,
  x8 = Bikeshare$windspeed,
  xS = Bikeshare$mnth
)
n_features <- 8
include_mdata <- seq_len(n_features + 1) # y + all predictors
index_mdata_xS <- n_features + 2 # xS

set.seed(19)

# Split data train test 80:20
n <- nrow(mdata_full)
shuffled_idx <- sample(n)
train_size <- floor(n * 0.8)
train_idx <- shuffled_idx[1:train_size]
test_idx <- shuffled_idx[(train_size + 1):n]

mdata_train_full <- mdata_full[train_idx, ]
mdata_test_full <- mdata_full[test_idx, ]

# Sort data by cluster labels
mdata_train_full <- mdata_train_full[order(mdata_train_full[, index_mdata_xS]), ]
mdata_test_full <- mdata_test_full[order(mdata_test_full[, index_mdata_xS]), ]

# mdata is mdata_full without the cluster labels (xS)
mdata_train <- mdata_train_full[, include_mdata]
mdata_test <- mdata_test_full[, include_mdata]

N_train <- nrow(mdata_train)
log_info("N_train: {N_train}")

N_test <- nrow(mdata_test)
log_info("N_test: {N_test}")

# Per cluster sizes in data p
mnth_sizes_train <- as.vector(table(mdata_train_full[, index_mdata_xS]))
mnth_sizes_test <- as.vector(table(mdata_test_full[, index_mdata_xS]))

# Plot train data
col_array_train <- array(NA, N_train)
offsets_train <- c(0, cumsum(mnth_sizes_train[-K]))
for (k in 1:K) {
  idx <- (offsets_train[k] + 1):(offsets_train[k] + mnth_sizes_train[k])
  col_array_train[idx] <- k
}

feature_names <- c(
  "bikers",
  "hr",
  "holiday",
  "weekday",
  "workingday",
  "weathersit",
  "temp",
  "hum",
  "windspeed"
)
plot_groups <- list(1:3, 4:6, 7:9)

for (pg in seq_along(plot_groups)) {
  par(mar = c(2, 5, 2, 2))
  par(mfrow = c(length(plot_groups[[pg]]), 1))
  for (j in plot_groups[[pg]]) {
    plot(mdata_train[, j], col = col_array_train, ylab = feature_names[j])
  }
  mtext("Train data (Bikeshare)", side = 3, line = -1.5, outer = TRUE)
}

M <- 250

# Split data into K clusters
mdata_train_k <- lapply(seq_len(K), function(k) {
  idx <- (offsets_train[k] + 1):(offsets_train[k] + mnth_sizes_train[k])
  mdata_train[idx, , drop = FALSE]
})
for (k in seq_len(K)) {
  log_info("Cluster {k} size: {nrow(mdata_train_k[[k]])}")
}

set.seed(7)

phi <- fn_shapley_cluster(
  K = K,
  M = M,
  data_train_k = mdata_train_k,
  data_test = mdata_test,
  prediction_accuracy = prediction_accuracy,
  method = mmethod
)

# Global Shapley values for each cluster
# phi has dimensions (N_test, K, M)
global_phi <- apply(phi, MARGIN = c(2, 3), FUN = mean, na.rm = TRUE)

full_prediction <- fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod)
log_info("MSE: {mean((full_prediction - mdata_test[, 1])^2)}")

# Plot convergence of Shapley values for each cluster
par(mar = c(5, 5.5, 3, 1))
par(mfrow = c(1, 1))
plot(
  seq_len(M),
  global_phi[1, ],
  type = "l",
  col = 1,
  lwd = 2,
  xlim = c(1, M),
  ylim = range(global_phi, na.rm = TRUE),
  xlab = "Number of iterations (M)",
  ylab = "Global Shapley values (Bikeshare)",
)
for (cluster_index in 2:K) {
  lines(seq_len(M), global_phi[cluster_index, ], col = cluster_index, lwd = 2)
}
legend(
  "topright",
  title = "Clusters included",
  legend = seq_len(K),
  col = seq_len(K),
  lty = 1,
  lwd = 2
)

# Pick 4 equally distributed test points
selected_points <- round(seq(1, N_test, length.out = 6))[2:5]

if (!prediction_accuracy) {
  plotted_value <- full_prediction
  plot_title <- "Predictions (Bikeshare)"
} else {
  plotted_value <- (full_prediction - mdata_test[, 1])^2
  plot_title <- "Squared Error (Bikeshare)"
}

par(mar = c(3, 3, 2, 2) * .7)
n_cols <- length(selected_points)
layout(
  matrix(
    c(
      rep(1, n_cols),
      2:(n_cols + 1),
      (n_cols + 2):(2 * n_cols + 1)
    ),
    nrow = 3,
    ncol = n_cols,
    byrow = TRUE
  ),
  respect = TRUE
)

# Top: full prediction or squared error
plot(seq_along(plotted_value), plotted_value, xaxs = "i", main = "", type = "l", col = "black", lwd = 3)
for (point_index in selected_points) {
  points(point_index, plotted_value[point_index], pch = 16, cex = 1.5, col = "red")
  abline(v = point_index, lty = 2)
}

# Middle: local Shapley values for each selected point
for (point_index in selected_points) {
  barplot(phi[point_index, , M], horiz = TRUE, col = seq_len(K), main = "")
  box()
}

# Bottom: convergence of Shapley values for each selected point
for (point_index in selected_points) {
  point_phi <- phi[point_index, , , drop = FALSE]
  plot(NA, xlim = c(1, M), ylim = range(point_phi, na.rm = TRUE), xlab = "", ylab = "", axes = FALSE)
  axis(1, labels = FALSE)
  axis(2)
  abline(h = 0, lty = 3)
  for (cluster_index in seq_len(K)) {
    lines(seq_len(M), phi[point_index, cluster_index, ], col = cluster_index)
  }
  box()
}

mtext(plot_title, side = 3, line = -7.5, outer = TRUE)
