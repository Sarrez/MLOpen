#Setup postgres for the database
FROM postgres:9.6
ENV POSTGRES_PASSWORD mlopen 
ENV POSTGRES_DB mlopen 
COPY docker/docker-db/init.sql /docker-entrypoint-initdb.d/
