all: air lintr

air:
    air format .

lintr:
    Rscript -e 'lintr::lint_dir("./src/custom")'

part_0:
    Rscript ./src/custom/main.R --prediction-accuracy false
    rm -rf ./figures/part_0
    mkdir -p ./figures/part_0
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_0/page

part_1:
    Rscript ./src/custom/main.R --prediction-accuracy true
    rm -rf ./figures/part_1
    mkdir -p ./figures/part_1
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_1/page

part_2:
    Rscript ./src/custom/main.R --prediction-accuracy true --global-classification true
    rm -rf ./figures/part_2
    mkdir -p ./figures/part_2
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_2/page

part_3:
    Rscript ./src/custom/main.R --prediction-accuracy true --method knn10
    rm -rf ./figures/part_3
    mkdir -p ./figures/part_3
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_3/page

part_4:
    Rscript ./src/custom/main.R --prediction-accuracy true --global-classification true --method knn10
    rm -rf ./figures/part_4
    mkdir -p ./figures/part_4
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_4/page

part_5:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy false
    rm -rf ./figures/part_5
    mkdir -p ./figures/part_5
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_5/page

part_6:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy true
    rm -rf ./figures/part_6
    mkdir -p ./figures/part_6
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_6/page

part_7:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy false --method knn10
    rm -rf ./figures/part_7
    mkdir -p ./figures/part_7
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_7/page

part_8:
    Rscript ./src/custom/Bikeshare.R --prediction-accuracy true --method knn10
    rm -rf ./figures/part_8
    mkdir -p ./figures/part_8
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_8/page
