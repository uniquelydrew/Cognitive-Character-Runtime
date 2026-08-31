FROM nginx@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10
# The entrypoint expands the server-side API credential into a tmpfs config. It
# never appears in browser-delivered JavaScript or in the immutable image.
COPY ui/nginx.conf /app/nginx.conf.template
COPY ui/start-nginx.sh /app/start-nginx.sh
COPY --chown=nginx:nginx ui/index.html /usr/share/nginx/html/index.html
COPY --chown=nginx:nginx ui/profiles.html /usr/share/nginx/html/profiles.html
COPY --chown=nginx:nginx ui/knowledge.html /usr/share/nginx/html/knowledge.html
COPY --chown=nginx:nginx ui/status.html /usr/share/nginx/html/status.html
RUN chmod 0555 /app/start-nginx.sh
USER nginx
ENTRYPOINT ["/app/start-nginx.sh"]
