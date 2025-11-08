.PHONY: env html clean

## env : creates and configures the environment
env:
	@if conda env list | grep -q '^ligo'; then \
		echo "Environment 'ligo' already exists. Updating..."; \
		conda env update -f environment.yml -n ligo; \
	else \
		echo "Creating environment 'ligo'..."; \
		conda env create -f environment.yml; \
	fi

## html : build the html rendering of the MyST site
html:
	myst build --html

## clean : clean up the figures, audio and _build folders
clean:
	rm -rf figures/*.png figures/*.jpg
	rm -rf audio/*.wav
	rm -rf _build
