library(R.utils)
library(logger)
library(optparse)
library(mlbench)


source("src/custom/common.R")

log_info("XAI Shapley Cluster - Boston Housing 2 Dataset")


# Init
mmethod <- "lm" # Which regression model to use
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

# Load BostonHousing2 dataset
data("BostonHousing2", package = "mlbench")

# Remove incomplete cases
BostonHousing2 <- BostonHousing2[complete.cases(BostonHousing2), ]
# write.csv(BostonHousing2, "datasets/BostonHousing2.csv", row.names = TRUE)

# Convert chas factor to numeric
BostonHousing2$chas <- as.numeric(BostonHousing2$chas) - 1

# Clusters by rad (index of accessibility to radial highways)
rad_labels <- sort(unique(BostonHousing2$rad))
K <- length(rad_labels)
log_info("Number of clusters: {K}")

# Build data matrix: y, x1..x12 (all predictors), xS (rad)
mdata_full <- cbind(
  y = BostonHousing2$cmedv,
  x1 = BostonHousing2$crim,
  x2 = BostonHousing2$zn,
  x3 = BostonHousing2$indus,
  x4 = BostonHousing2$chas,
  x5 = BostonHousing2$nox,
  x6 = BostonHousing2$rm,
  x7 = BostonHousing2$age,
  x8 = BostonHousing2$dis,
  x9 = BostonHousing2$tax,
  x10 = BostonHousing2$ptratio,
  x11 = BostonHousing2$b,
  x12 = BostonHousing2$lstat,
  xS = BostonHousing2$rad
)
n_features <- 12
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
rad_sizes_train <- as.vector(table(mdata_train_full[, index_mdata_xS]))
rad_sizes_test <- as.vector(table(mdata_test_full[, index_mdata_xS]))


# Plot train data
col_array_train <- array(NA, N_train)
offsets_train <- c(0, cumsum(rad_sizes_train[-K]))
for (k in 1:K) {
  idx <- (offsets_train[k] + 1):(offsets_train[k] + rad_sizes_train[k])
  col_array_train[idx] <- k
}

par(mar = c(2, 5, 2, 2))

plot_indices <- c(2, 3, 4) # x1, x2, x3
par(mfrow = c(length(plot_indices), 1))

feature_names <- c("crim", "zn", "indus")
for (j in seq_along(plot_indices)) {
  plot(mdata_train[, plot_indices[j]], col = col_array_train, ylab = feature_names[j])
}

mtext("Train data (BostonHousing2)", side = 3, line = -1.5, outer = TRUE)


set.seed(21)

M <- 250

# Split data into K clusters
mdata_train_k <- lapply(seq_len(K), function(k) {
  idx <- (offsets_train[k] + 1):(offsets_train[k] + rad_sizes_train[k])
  mdata_train[idx, , drop = FALSE]
})

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
  ylab = "Global Shapley values (BostonHousing2)",
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
full_prediction <- fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod)

if (!prediction_accuracy) {
  plotted_value <- full_prediction
  plot_title <- "Predictions (BostonHousing2)"
} else {
  plotted_value <- (full_prediction - mdata_test[, 1])^2
  plot_title <- "Squared Error (BostonHousing2)"
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
plot(seq_along(plotted_value), plotted_value, xaxs = "i", main = "", type = "l", col = 11, lwd = 3)
for (point_index in selected_points) {
  points(point_index, plotted_value[point_index], pch = 16, cex = 1.5, col = 9)
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
