FROM nginx:stable-alpine

# Remove default nginx content and copy frontend static files
RUN rm -rf /usr/share/nginx/html/*
COPY frontend/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
