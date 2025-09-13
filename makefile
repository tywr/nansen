CONTAINER = nansen

build-container:
	docker compose build ${CONTAINER}

.SILENT:
build-package: build-container
	docker compose run --rm ${CONTAINER}

.SILENT:
test-package: build-package
	docker compose run --rm ${CONTAINER} python


.SILENT:
publish: build-package
	twine upload dist/*
