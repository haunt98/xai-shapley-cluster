all: air

air:
    air format .

lintr:
    Rscript -e 'lintr::lint_dir("./src/custom")'

part_0:
    Rscript ./src/custom/main.R --prediction-accuracy false --global-classification false
    rm -rf ./figures/part_0
    mkdir -p ./figures/part_0
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_0/page

part_1:
    Rscript ./src/custom/main.R --prediction-accuracy true --global-classification false
    rm -rf ./figures/part_1
    mkdir -p ./figures/part_1
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_1/page

part_2:
    Rscript ./src/custom/main.R --prediction-accuracy true --global-classification true
    rm -rf ./figures/part_2
    mkdir -p ./figures/part_2
    pdftoppm -png -r 300 ./Rplots.pdf ./figures/part_2/page
