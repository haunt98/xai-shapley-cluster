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

# Balance clusters: subsample each month to the smallest month size
month_labels <- sort(unique(airquality$Month))
K <- length(month_labels)

month_counts <- table(airquality$Month)
min_count <- min(month_counts)

set.seed(2)
balanced <- do.call(
  rbind,
  lapply(month_labels, function(m) {
    subset <- airquality[airquality$Month == m, ]
    subset[sample(nrow(subset), min_count), ]
  })
)

# Sort by Month so rows are contiguous per cluster
balanced <- balanced[order(balanced$Month), ]

# Build data matrix: y, x1, x2, x3, xS (month)
mdata_full <- cbind(
  y = balanced$Ozone,
  x1 = balanced$Solar.R,
  x2 = balanced$Wind,
  x3 = balanced$Temp,
  xS = balanced$Month
)
include_mdata <- c(1, 2, 3, 4) # y, x1, x2, x3
index_mdata_xS <- 5 # xS

# Per-cluster split: first 4 rows per month for train, last 5 for eval
K_train <- 4
K_eval <- min_count - K_train

N_train <- K_train * K
N_eval <- K_eval * K

train_indices <- as.vector(sapply(seq_len(K), function(k) {
  ((k - 1) * min_count + 1):((k - 1) * min_count + K_train)
}))
eval_indices <- as.vector(sapply(seq_len(K), function(k) {
  ((k - 1) * min_count + K_train + 1):(k * min_count)
}))

mdata_train_full <- mdata_full[train_indices, ]
mdata_eval_full <- mdata_full[eval_indices, ]

# mdata without cluster labels
mdata <- rbind(mdata_train_full, mdata_eval_full)
mdata <- mdata[, include_mdata]

mdata_train <- mdata[1:N_train, ]
mdata_eval <- mdata[(N_train + 1):(N_train + N_eval), ]

N_train <- dim(mdata_train)[1]
N_eval <- dim(mdata_eval)[1]

mdata_test <- mdata_eval
N_test <- N_eval

J <- dim(mdata)[2] - 1 # number of features (3: Solar.R, Wind, Temp)

K_test <- K_eval

# Plot train data
col_array_train <- array(NA, N_train)
for (k in 1:K) {
  col_array_train[((k - 1) * K_train + 1):(K_train * k)] <- rep(k, K_train)
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

# Split data into K clusters
mdata_train_k <- array(NA, dim = c(K_train, J + 1, K))
mdata_test_k <- array(NA, dim = c(K_test, J + 1, K))
mdata_eval_k <- array(NA, dim = c(K_eval, J + 1, K))

for (k in 1:K) {
  for (j in 1:(J + 1)) {
    mdata_train_k[, j, k] <- mdata_train[(K_train * (k - 1) + 1):(K_train * k), j]
    mdata_test_k[, j, k] <- mdata_test[(K_test * (k - 1) + 1):(K_test * k), j]
    mdata_eval_k[, j, k] <- mdata_eval[(K_eval * (k - 1) + 1):(K_eval * k), j]
  }
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
