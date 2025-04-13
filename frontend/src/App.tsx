import { useState, useEffect, useCallback, ChangeEvent, FormEvent } from 'react'
import { Routes, Route, Link } from 'react-router-dom' // Возвращаем роутинг
import axios from 'axios'
import LineDetailPage from './LineDetailPage' // Импортируем новую страницу

// --- Иконки SVG (оставляем как есть) ---
const IconGear = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 1.255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-1.255c.007-.378-.137-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const IconRefresh = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-2 animate-spin">
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
  </svg>
);

const IconQuestion = () => (
 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
  </svg>
);

// Убрали IconTrend

// Интерфейс для ответа от /api/get (используется на главной)
interface LineStatusResponse {
  area_name: string;
  last_update: string | null; 
  status: string;
  data: Record<string, any>[]; 
}

const API_BASE_URL = 'http://localhost:8000';

// --- Компонент основной страницы (Dashboard) --- 
function DashboardPage() {
  const [lines, setLines] = useState<LineStatusResponse[]>([]); // Используем тот же интерфейс, но `data` не будет отображаться здесь
  const [isLoading, setIsLoading] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newAreaName, setNewAreaName] = useState('');
  const [newFilePath, setNewFilePath] = useState(''); 

  // Функция для загрузки списка линий (без полных данных)
  // Можно было бы создать отдельный эндпоинт /api/get/summary, но пока используем /api/get
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<LineStatusResponse[]>(`${API_BASE_URL}/api/get`);
      // Сохраняем все, но LineCard будет использовать только name, status, last_update
      setLines(response.data);
    } catch (err: any) {
      console.error('Ошибка загрузки данных линий:', err);
      setError(err.response?.data?.detail || err.message || 'Не удалось загрузить данные линий.');
      setLines([]);
      } finally {
      setIsLoading(false);
    }
  }, []);

  // Загрузка данных при первом рендере и интервал обновления
  useEffect(() => {
    fetchData();
    const intervalId = setInterval(fetchData, 60 * 1000);
    return () => clearInterval(intervalId);
  }, [fetchData]);

  // Обработчик добавления новой линии (без изменений)
  const handleAddLine = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!newAreaName || !newFilePath) {
      setError('Необходимо указать название линии и путь к файлу.');
      return;
    }
    setIsAdding(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/add`, {
         area_name: newAreaName,
         file_path: newFilePath,
       });
      console.log('Ответ от сервера /api/add:', response.data);
      setNewAreaName('');
      setNewFilePath('');
      setShowAddForm(false);
      await fetchData(); // Обновляем список
    } catch (err: any) {
      console.error('Ошибка добавления линии:', err);
      setError(err.response?.data?.detail || err.message || 'Не удалось добавить линию.');
    } finally {
      setIsAdding(false);
    }
  };

  // --- Компонент карточки линии (теперь это ссылка) --- 
  const LineCard = ({ line }: { line: LineStatusResponse }) => { // Больше не нужен onDeleted напрямую
    const [isDeleting, setIsDeleting] = useState(false);
    // Убрали showData

    const handleDelete = async (e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault(); // ВАЖНО: Предотвращаем переход по ссылке при клике на кнопку удаления
      e.stopPropagation(); 
      if (isDeleting) return;

      if (window.confirm(`Вы уверены, что хотите удалить линию "${line.area_name}"? Это действие необратимо.`)) {
        setIsDeleting(true);
        setError(null); 
        try {
          await axios.delete(`${API_BASE_URL}/api/lines/${line.area_name}`);
          fetchData(); // Обновляем список линий после удаления
        } catch (err: any) {
          console.error(`Ошибка удаления линии ${line.area_name}:`, err);
          setError(err.response?.data?.detail || err.message || `Не удалось удалить линию ${line.area_name}.`);
        } finally {
          setIsDeleting(false);
        }
      }
    };

    // Иконка и цвет статуса (без изменений)
    let StatusIcon = IconQuestion; 
    let statusColorClass = 'text-gray-500';
    if (line.status === 'OK') {
        StatusIcon = IconGear;
        statusColorClass = 'text-green-600';
    } else if (line.status.startsWith('Ошибка')) {
        StatusIcon = IconQuestion;
        statusColorClass = 'text-red-600';
    }

    return (
      // Оборачиваем всю карточку в Link, КРОМЕ кнопки удаления
      <div className="relative flex flex-col border rounded-lg shadow-sm overflow-hidden bg-card group">
          <Link 
              to={`/area/${line.area_name}`} 
              className="block p-4 border-b bg-gray-50 hover:bg-gray-100 transition-colors duration-150" // Стилизуем ссылку
            >
              <h2 className="text-lg font-semibold text-card-foreground pr-8">{line.area_name}</h2> { /* Увеличили отступ справа под кнопку */}
              <div className={`flex items-center mt-1 text-sm ${statusColorClass}`}>
                  <StatusIcon />
                  <span>{line.status}</span>
              </div>
              {line.last_update && (
                  <p className="text-xs text-gray-500 mt-1">
                      Обновлено: {new Date(line.last_update).toLocaleString()}
                  </p>
              )}
          </Link>
           {/* Кнопка удаления остается НАД ссылкой */}
           <button
              onClick={handleDelete}
              disabled={isDeleting}
              className={`absolute top-3 right-3 z-10 p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-100 rounded-full transition-colors duration-150 ${isDeleting ? 'opacity-50 cursor-not-allowed' : ''}`}
              aria-label={`Удалить ${line.area_name}`}
              title={`Удалить ${line.area_name}`}
          >
               {/* SVG иконки удаления */} 
               {isDeleting ? (
                  <svg className="w-5 h-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
               ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                  </svg>
               )}
           </button>
          {/* Убрали контейнер с данными и кнопку "Показать/Скрыть" */} 
      </div>
    );
  };

  // --- Рендер основной страницы --- 
  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      {/* Header (без изменений) */} 
      <header className="sticky top-0 z-20 border-b bg-white shadow-sm">
        <div className="container flex items-center justify-between h-16 px-4 mx-auto md:px-6">
          <h1 className="text-xl font-semibold">Мониторинг Производственных Линий</h1>
          <button
            onClick={fetchData} 
            disabled={isLoading || isAdding}
            className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium transition-colors border border-gray-300 rounded-md shadow-sm bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
          >
            {isLoading ? (
               <><IconRefresh /> Обновление...</>
            ) : 'Обновить вручную'}
          </button>
        </div>
      </header>

      {/* Main (без изменений в форме добавления, только в списке) */} 
      <main className="container px-4 py-8 mx-auto md:px-6">
        {error && (
          <div className="relative w-full p-4 mb-6 text-sm text-red-700 bg-red-100 border border-red-300 rounded-md" role="alert">
            <strong className="font-semibold">Ошибка: </strong>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="absolute top-2.5 right-2.5 p-1.5 rounded-md text-red-500 hover:text-red-700 hover:bg-red-200 transition-colors" aria-label="Закрыть">
               <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                 <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
               </svg>
            </button>
          </div>
        )}

        {!showAddForm ? (
           <div className="mb-8">
             <button
               onClick={() => setShowAddForm(true)}
               className="flex items-center justify-center w-full px-6 py-5 text-lg font-medium text-center text-gray-500 transition-colors border-2 border-gray-300 border-dashed rounded-lg hover:border-gray-400 hover:text-gray-600 hover:bg-gray-50"
             >
               <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                 <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
               </svg>
               Добавить производственную линию
             </button>
          </div>
        ) : (
          <form onSubmit={handleAddLine} className="p-6 mb-8 bg-white border border-gray-200 rounded-lg shadow-sm">
             {/* Форма добавления без изменений */}
             <h2 className="mb-5 text-xl font-semibold">Добавить новую линию</h2>
             <div className="mb-4">
               <label htmlFor="areaName" className="block mb-1.5 text-sm font-medium text-gray-700">Название линии (Table Name)</label>
               <input type="text" id="areaName" value={newAreaName} onChange={(e) => setNewAreaName(e.target.value)} placeholder="Например, production_line_1" required className="block w-full px-3 py-2 text-sm border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" />
             </div>
             <div className="mb-5">
               <label htmlFor="filePath" className="block mb-1.5 text-sm font-medium text-gray-700">Путь к CSV файлу</label>
               <input type="text" id="filePath" value={newFilePath} onChange={(e) => setNewFilePath(e.target.value)} placeholder="Например, C:\data\line1_log.csv или /home/user/data/line1.csv" required className="block w-full px-3 py-2 text-sm border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" />
             </div>
             <div className="flex items-center justify-end space-x-3">
               <button type="button" onClick={() => { setShowAddForm(false); setError(null); }} className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50" disabled={isAdding}>Отмена</button>
               <button type="submit" disabled={isAdding} className="inline-flex justify-center px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50">{isAdding ? 'Добавление...' : 'Добавить'}</button>
          </div>
           </form>
        )}

        {/* Отображение списка линий (карточки теперь ссылки) */} 
        <h2 className="text-xl font-semibold mb-4">Активные линии</h2>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
           {/* Skeleton loader (без изменений) */} 
          {isLoading && lines.length === 0 && (
             Array.from({ length: 3 }).map((_, index) => (
               <div key={index} className="flex flex-col border rounded-lg shadow-sm animate-pulse bg-white">
                 <div className="p-4 border-b bg-gray-100">
                    <div className="h-5 bg-gray-300 rounded w-3/4 mb-2"></div>
                    <div className="h-4 bg-gray-300 rounded w-1/2 mb-1"></div>
                    <div className="h-3 bg-gray-300 rounded w-1/3"></div>
                 </div>
                 {/* Убрали placeholder данных */}
               </div>
             ))
          )}
          {!isLoading && lines.length === 0 && !error && <p className="col-span-full text-center text-gray-500">Нет активных производственных линий для отображения.</p>}
          {lines.map((line) => (
            <LineCard key={line.area_name} line={line} /> // Передаем только line
          ))}
        </div>
      </main>
    </div>
  );
}

// --- Основной компонент App с маршрутизацией --- 
function App() {
  return (
    <Routes>
      {/* Главная страница со списком */} 
      <Route path="/" element={<DashboardPage />} /> 
      {/* Страница деталей линии */} 
      <Route path="/area/:areaName" element={<LineDetailPage />} />
      {/* Можно добавить Route для 404 страницы */}
    </Routes>
  );
}

export default App; 