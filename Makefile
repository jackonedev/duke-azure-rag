format:
	isort --profile=black --skip .venv --skip .env . &&\
	autopep8 --in-place webapp/*.py &&\
	black --line-length 90 . --exclude '(\.venv|\.env)'

lint:
	pylint --disable=R,C webapp/*.py

