all: air

air:
    air format .

lintr:
    Rscript -e 'lintr::lint_dir("./src/custom")'
