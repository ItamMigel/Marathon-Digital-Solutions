# Используем базовый образ Node.js (укажи нужную версию, если она отличается)
FROM node:18-alpine

# Устанавливаем рабочую директорию
WORKDIR /app/frontend

# Копируем package.json и package-lock.json (или yarn.lock)
COPY package*.json ./

# Устанавливаем зависимости
RUN npm install

# Копируем остальной код фронтенда
COPY . .

# Открываем порт, на котором работает dev server (стандартный для Vite/Next.js - 5173)
EXPOSE 5173

# Команда для запуска dev server
CMD ["npm", "run", "dev"] 