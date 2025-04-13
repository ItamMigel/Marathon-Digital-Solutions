import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

// Интерфейс для данных одной строки из CSV (примерный, нужно будет адаптировать)
interface DataRecord {
  timestamp: string; // Или Date
  // Другие поля из вашего CSV
  value1: number;
  value2: string;
  // ...
}

// Интерфейс для информации о производственной линии от /api/get
interface ProductionLine {
  area_name: string;
  last_update: string; // Или Date
  status: string;
  data: DataRecord[];
}

const API_BASE_URL = 'http://localhost:8000/api'; // Базовый URL вашего бэкенда

function App() {
  const [lines, setLines] = useState<ProductionLine[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAddForm, setShowAddForm] = useState(false)
  const [newAreaName, setNewAreaName] = useState('')
  const [newFilePath, setNewFilePath] = useState('')

  // Функция для загрузки данных о линиях
  const fetchData = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await axios.get<ProductionLine[]>(`${API_BASE_URL}/get`)
      setLines(response.data)
    } catch (err) {
      console.error('Ошибка загрузки данных:', err)
      setError('Не удалось загрузить данные о производственных линиях.')
      // Опционально: очистить старые данные при ошибке
      // setLines([]);
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Загрузка данных при первом рендере и установка интервала автообновления
  useEffect(() => {
    fetchData() // Загружаем данные сразу
    const intervalId = setInterval(fetchData, 60 * 1000) // Обновляем каждую минуту

    // Очистка интервала при размонтировании компонента
    return () => clearInterval(intervalId)
  }, [fetchData])

  // Обработчик добавления новой линии
  const handleAddLine = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!newAreaName || !newFilePath) {
      setError('Необходимо указать название линии и путь к файлу.')
      return
    }
    setIsLoading(true)
    setError(null)
    try {
      await axios.post(`${API_BASE_URL}/add`, {
        area_name: newAreaName,
        file_path: newFilePath,
      })
      setNewAreaName('')
      setNewFilePath('')
      setShowAddForm(false)
      await fetchData() // Обновляем данные после добавления
    } catch (err: any) {
      console.error('Ошибка добавления линии:', err)
      setError(err.response?.data?.error || 'Не удалось добавить линию.')
    } finally {
      setIsLoading(false)
    }
  }

  // --- Иконки как SVG компоненты ---
  const IconGear = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 mx-auto mb-2 text-gray-600">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 1.255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-1.255c.007-.378-.137-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );

  const IconRefresh = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 mx-auto mb-2 text-gray-600 animate-spin">
      <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
    </svg>
  );

  const IconQuestion = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 mx-auto mb-2 text-gray-600">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
    </svg>
  );

  const IconTrend = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-full h-16 mx-auto text-gray-700">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941" />
    </svg>
  );

  // --- Компонент для отображения одной линии (переделанный) ---
  const LineCard = ({ line }: { line: ProductionLine }) => {

    let StatusIcon;
    let statusText = line.status; // По умолчанию используем статус с бэкенда
    let statusColor = 'text-gray-600'; // Цвет текста по умолчанию
    let secondaryText = ''; // Дополнительный текст для ошибки

    switch (line.status.toUpperCase()) { // Приводим к верхнему регистру для надежности
      case 'OK':
      case 'RUNNING': // Допустим, 'RUNNING' тоже считается нормальным состоянием
        StatusIcon = IconGear;
        statusText = 'Всё работает';
        statusColor = 'text-green-700';
        break;
      case 'UPDATING':
        StatusIcon = IconRefresh;
        statusText = 'Обновление статуса...';
        statusColor = 'text-blue-700';
        break;
      case 'ERROR':
      case 'PROBLEM':
      default: // Все остальные статусы считаем проблемой
        StatusIcon = IconQuestion;
        statusText = 'Проблема';
        statusColor = 'text-red-700';
        // Пытаемся извлечь доп. информацию, если она есть (пример)
        // Возможно, ваш статус содержит детали после двоеточия?
        if (line.status.includes(':')) {
          secondaryText = line.status.split(':')[1].trim();
        } else if (line.status.toUpperCase() !== 'ERROR' && line.status.toUpperCase() !== 'PROBLEM') {
           secondaryText = line.status; // Показать оригинальный статус если он не стандартный "ERROR"/"PROBLEM"
        }
        break;
    }

    return (
      <div className="flex flex-col text-center border border-gray-300 rounded-lg shadow-sm">
        <h2 className="px-4 py-2 text-base font-semibold border-b border-gray-300">{line.area_name}</h2>
        <div className="flex flex-col justify-between flex-grow p-4 bg-gray-100">
          <div className="mb-4">
            <StatusIcon />
            <p className={`text-lg font-bold ${statusColor}`}>{statusText}</p>
            {secondaryText && (
              <p className="mt-1 text-xs text-gray-500">{secondaryText}</p>
            )}
          </div>
          <IconTrend />
        </div>
        {/* Убрана информация о последнем обновлении - не видно на макете */}
        {/* Убрана таблица данных */}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="container flex items-center justify-between h-16 mx-auto">
          <h1 className="text-xl font-bold">Мониторинг Производственных Линий</h1>
          <button
            onClick={() => fetchData()}
            disabled={isLoading}
            className="px-4 py-2 text-sm border rounded-md hover:bg-accent disabled:opacity-50"
          >
            {isLoading ? 'Обновление...' : 'Обновить вручную'}
          </button>
        </div>
      </header>

      <main className="container py-6 mx-auto">
        {error && (
          <div className="p-4 mb-6 text-red-800 bg-red-100 border border-red-300 rounded-md relative" role="alert">
            <strong className="font-bold">Ошибка:</strong>
            <span className="block sm:inline"> {error}</span>
            <button onClick={() => setError(null)} className="absolute top-0 bottom-0 right-0 px-4 py-3" aria-label="Закрыть">
              <svg className="w-6 h-6 text-red-500 fill-current" role="button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><title>Закрыть</title><path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.2 1.2 0 0 1 0 1.698z"/></svg>
            </button>
          </div>
        )}

        {!showAddForm ? (
          <div className="mb-6">
            <button
              onClick={() => setShowAddForm(true)}
              className="flex items-center justify-center w-full px-6 py-4 text-lg text-center text-gray-500 border-2 border-gray-300 border-dashed rounded-lg hover:border-gray-400 hover:text-gray-600 bg-gray-50"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
              Добавить производственную линию
            </button>
          </div>
        ) : (
          <form onSubmit={handleAddLine} className="p-6 mb-6 border rounded-lg shadow bg-card">
            <h2 className="mb-4 text-xl font-semibold">Добавить новую линию</h2>
            <div className="mb-4">
              <label htmlFor="areaName" className="block mb-1 text-sm font-medium text-muted-foreground">Название линии</label>
              <input
                type="text"
                id="areaName"
                value={newAreaName}
                onChange={(e) => setNewAreaName(e.target.value)}
                placeholder="Например, Линия Упаковки #1"
                required
                className="w-full p-2 border rounded-md bg-input text-foreground focus:ring-primary focus:border-primary"
              />
            </div>
            <div className="mb-4">
              <label htmlFor="filePath" className="block mb-1 text-sm font-medium text-muted-foreground">Путь к CSV файлу</label>
              <input
                type="text" // Пока оставляем текстовое поле, можно заменить на type="file"
                id="filePath"
                value={newFilePath}
                onChange={(e) => setNewFilePath(e.target.value)}
                placeholder="Например, C:\data\line1_log.csv или /home/user/data/line1.csv"
                required
                className="w-full p-2 border rounded-md bg-input text-foreground focus:ring-primary focus:border-primary"
              />
               {/* TODO: Реализовать выбор файла через <input type="file"> если нужно */}
            </div>
            <div className="flex items-center justify-end space-x-3">
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 text-sm border rounded-md hover:bg-muted"
                disabled={isLoading}
              >
                Отмена
              </button>
              <button
                type="submit"
                disabled={isLoading}
                className="px-4 py-2 text-sm text-white rounded-md bg-primary hover:bg-primary/90 disabled:opacity-50"
              >
                {isLoading ? 'Добавление...' : 'Добавить'}
              </button>
            </div>
          </form>
        )}

        {/* Отображение списка линий в виде сетки */}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {isLoading && lines.length === 0 && (
             Array.from({ length: 3 }).map((_, index) => (
                <div key={index} className="flex flex-col text-center border border-gray-300 rounded-lg shadow-sm animate-pulse">
                  <div className="h-10 px-4 py-2 bg-gray-200 border-b border-gray-300 rounded-t-lg"></div>
                  <div className="flex flex-col justify-between flex-grow p-4 bg-gray-100 rounded-b-lg">
                    <div className="mb-4">
                      <div className="w-12 h-12 mx-auto mb-2 bg-gray-300 rounded-full"></div>
                      <div className="h-6 mb-1 bg-gray-300 rounded w-3/4 mx-auto"></div>
                      <div className="h-4 bg-gray-300 rounded w-1/2 mx-auto"></div>
                    </div>
                    <div className="w-full h-16 bg-gray-300 rounded"></div>
                  </div>
                </div>
             ))
          )}
          {!isLoading && lines.length === 0 && !error && <p className="col-span-full text-center text-muted-foreground">Нет активных производственных линий для отображения.</p>}
          {lines.map((line) => (
            <LineCard key={line.area_name} line={line} />
          ))}
        </div>
      </main>
    </div>
  )
}

export default App 