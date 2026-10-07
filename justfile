all: air lintr

air:
    air format .

lintr:
    Rscript -e 'lintr::lint_dir("./src/custom")'

parts: parts_0_4 parts_5_8

parts_0_4: part_0 part_1 part_2 part_3 part_4

parts_5_8: part_5 part_6 part_7 part_8

part_0:
    Rscript ./src/custom/main.R --prediction-accuracy false --output ./results/part_0 --pdf ./results/part_0/plots.pdf
    rm -rf ./figures/part_0
    mkdir -p ./figures/part_0
    pdftoppm -png -r 300 ./results/part_0/plots.pdf ./figures/part_0/page

part_1:
    Rscript ./src/custom/main.R --prediction-accuracy false --method knn10 --output ./results/part_1 --pdf ./results/part_1/plots.pdf
    rm -rf ./figures/part_1
    mkdir -p ./figures/part_1
    pdftoppm -png -r 300 ./results/part_1/plots.pdf ./figures/part_1/page

part_2:
    Rscript ./src/custom/main.R --prediction-accuracy true --output ./results/part_2 --pdf ./results/part_2/plots.pdf
    rm -rf ./figures/part_2
    mkdir -p ./figures/part_2
    pdftoppm -png -r 300 ./results/part_2/plots.pdf ./figures/part_2/page

part_3:
    Rscript ./src/custom/main.R --prediction-accuracy true --method knn10 --output ./results/part_3 --pdf ./results/part_3/plots.pdf
    rm -rf ./figures/part_3
    mkdir -p ./figures/part_3
    pdftoppm -png -r 300 ./results/part_3/plots.pdf ./figures/part_3/page

part_4:
    Rscript ./src/custom/main.R --prediction-accuracy true --global-classification true --output ./results/part_4 --pdf ./results/part_4/plots.pdf
    rm -rf ./figures/part_4
    mkdir -p ./figures/part_4
    pdftoppm -png -r 300 ./results/part_4/plots.pdf ./figures/part_4/page

part_5:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy false --output ./results/part_5 --pdf ./results/part_5/plots.pdf
    rm -rf ./figures/part_5
    mkdir -p ./figures/part_5
    pdftoppm -png -r 300 ./results/part_5/plots.pdf ./figures/part_5/page

part_6:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy true --output ./results/part_6 --pdf ./results/part_6/plots.pdf
    rm -rf ./figures/part_6
    mkdir -p ./figures/part_6
    pdftoppm -png -r 300 ./results/part_6/plots.pdf ./figures/part_6/page

part_7:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy false --method knn10 --output ./results/part_7 --pdf ./results/part_7/plots.pdf
    rm -rf ./figures/part_7
    mkdir -p ./figures/part_7
    pdftoppm -png -r 300 ./results/part_7/plots.pdf ./figures/part_7/page

part_8:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy true --method knn10 --output ./results/part_8 --pdf ./results/part_8/plots.pdf
    rm -rf ./figures/part_8
    mkdir -p ./figures/part_8
    pdftoppm -png -r 300 ./results/part_8/plots.pdf ./figures/part_8/page
