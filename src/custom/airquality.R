library(R.utils)
library(logger)
library(optparse)


source("src/custom/common.R")

log_info("XAI Shapley Cluster - Airquality Dataset")

# Init
mmethod <- "lm0" # Which regression model to use

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


# Load airquality dataset
data(airquality)

# Remove incomplete cases
airquality <- airquality[complete.cases(airquality), ]
write.csv(airquality, "datasets/airquality.csv", row.names = TRUE)

# Clusters by month, keep original sizes (no balancing)
month_labels <- sort(unique(airquality$Month))
K <- length(month_labels)

# Sort by Month so rows are contiguous per cluster
airquality <- airquality[order(airquality$Month), ]
month_sizes <- as.vector(table(airquality$Month))

# Build data matrix: y, x1, x2, x3, xS (month)
mdata_full <- cbind(
  y = airquality$Ozone,
  x1 = airquality$Solar.R,
  x2 = airquality$Wind,
  x3 = airquality$Temp,
  xS = airquality$Month
)
include_mdata <- c(1, 2, 3, 4) # y, x1, x2, x3
index_mdata_xS <- 5 # xS

# Per-cluster split: first 4 rows per month for train, rest for eval
K_train <- 4

offsets <- c(0, cumsum(month_sizes[-K]))

train_indices <- unlist(lapply(seq_len(K), function(k) {
  offsets[k] + seq_len(K_train)
}))
test_indices <- unlist(lapply(seq_len(K), function(k) {
  offsets[k] + (K_train + 1):month_sizes[k]
}))

mdata_train_full <- mdata_full[train_indices, ]
mdata_test_full <- mdata_full[test_indices, ]

# mdata without cluster labels
mdata <- rbind(mdata_train_full, mdata_test_full)
mdata <- mdata[, include_mdata]

mdata_train <- mdata[1:length(train_indices), ]
mdata_test <- mdata[(length(train_indices) + 1):(length(train_indices) + length(test_indices)), ]

N_train <- dim(mdata_train)[1]
N_test <- dim(mdata_test)[1]

J <- dim(mdata)[2] - 1 # number of features (3: Solar.R, Wind, Temp)

# Plot train data
col_array_train <- array(NA, N_train)
for (k in 1:K) {
  col_array_train[(K_train * (k - 1) + 1):(K_train * k)] <- rep(k, K_train)
}

par(mar = c(2, 5, 2, 2))
par(mfrow = c(length(include_mdata), 1))

feature_names <- c("Ozone", "Solar.R", "Wind", "Temp")
for (j in include_mdata) {
  plot(mdata_train[, j], col = col_array_train, ylab = feature_names[j])
}
mtext("Train data (airquality)", side = 3, line = -1.5, outer = TRUE)

set.seed(21)

M <- 250

# Split data into K clusters as a list (each cluster has different size)
mdata_train_k <- lapply(seq_len(K), function(k) {
  mdata_train[(K_train * (k - 1) + 1):(K_train * k), , drop = FALSE]
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

# Plot results
selected_points <- round(seq(5, N_test - 5, length.out = 5))
full_prediction <- fn_prediction(data_train = mdata_train, data_test = mdata_test, method = mmethod)

plotted_value <- (full_prediction - mdata_test[, 1])^2
plot_title <- "Squared Error (airquality)"

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

# Top: squared error
plot(seq_along(plotted_value), plotted_value, xaxs = "i", main = "", type = "l", col = 11, lwd = 3)
for (point_index in selected_points) {
  points(point_index, plotted_value[point_index], pch = 16, cex = 1.5, col = 9)
  abline(v = point_index, lty = 2)
}

# Middle: local Shapley values (final iteration)
for (point_index in selected_points) {
  barplot(phi[point_index, , M], horiz = TRUE, col = seq_len(K), main = paste("Month", month_labels))
  box()
}

# Bottom: convergence
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

mtext(plot_title, side = 3, line = -11.5, outer = TRUE)
