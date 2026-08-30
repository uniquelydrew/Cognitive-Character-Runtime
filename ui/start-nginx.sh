#!/bin/sh
set -eu

: "${API_AUTH_TOKEN:?API_AUTH_TOKEN is required}"
envsubst '${API_AUTH_TOKEN}' < /app/nginx.conf.template > /tmp/nginx.conf
exec nginx -c /tmp/nginx.conf -g 'daemon off;'
