import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
// Импортируем Recharts для графика аналитики
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// --- Иконки SVG (можно вынести в отдельный файл, если используются еще где-то) ---
const IconGear = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-1.5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 1.255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-1.255c.007-.378-.137-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const IconQuestion = () => (
 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-1.5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
  </svg>
);

const IconArrowLeft = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-1">
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
  </svg>
);

// Добавим иконку для ссылки на аналитику
const IconChartBarSquare = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-1.5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />
  </svg>
);

// Интерфейс для информации о пагинации (должен совпадать с Pydantic моделью)
interface PaginationInfo {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

// Интерфейс для детального ответа от /api/lines/{areaName} (с пагинацией)
interface LineDetailResponse {
  area_name: string;
  last_update: string | null;
  status: string;
  data: Record<string, any>[];
  pagination: PaginationInfo;
}

const API_BASE_URL = 'http://localhost:8000';
const DEFAULT_PAGE_SIZE = 20; // Сколько элементов на странице по умолчанию

function LineDetailPage() {
  const { areaName } = useParams<{ areaName: string }>(); 
  const [lineData, setLineData] = useState<LineDetailResponse | null>(null); // Храним весь ответ, включая пагинацию
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1); // Состояние для текущей страницы

  // Функция для загрузки данных конкретной линии с учетом страницы
  const fetchLineData = useCallback(async (pageToLoad: number) => {
    if (!areaName) return; 
    setIsLoading(true);
    setError(null);
    try {
      // Передаем параметры page и page_size
      const response = await axios.get<LineDetailResponse>(
        `${API_BASE_URL}/api/lines/${areaName}`, 
        {
           params: { 
              page: pageToLoad,
              page_size: DEFAULT_PAGE_SIZE 
           }
        }
      );
      setLineData(response.data);
      setCurrentPage(response.data.pagination.page); // Обновляем текущую страницу из ответа
    } catch (err: any) {
      console.error(`Ошибка загрузки данных для линии ${areaName} (страница ${pageToLoad}):`, err);
      setError(err.response?.data?.detail || err.message || 'Не удалось загрузить данные линии.');
      setLineData(null); // Сбрасываем данные при ошибке
      setCurrentPage(1); // Сбрасываем страницу на 1
    } finally {
      setIsLoading(false);
    }
  }, [areaName]); // Зависимость только от areaName

  // Загрузка данных при монтировании компонента и установка интервала обновления
  useEffect(() => {
    fetchLineData(currentPage); // Загружаем текущую страницу при монтировании или изменении areaName
    
    // Устанавливаем интервал для периодического обновления
    const intervalId = setInterval(() => {
        // Перезагружаем данные для ТЕКУЩЕЙ страницы
        fetchLineData(currentPage);
    }, 60 * 1000); // Обновляем каждую минуту

    // Очистка интервала при размонтировании компонента
    return () => clearInterval(intervalId);

  }, [fetchLineData, currentPage]); // Добавляем currentPage в зависимости, чтобы интервал перезапускался с правильной страницей

  // Обработчики для кнопок пагинации
  const handlePrevPage = () => {
    if (lineData && lineData.pagination.page > 1) {
      fetchLineData(lineData.pagination.page - 1);
    }
  };

  const handleNextPage = () => {
    if (lineData && lineData.pagination.page < lineData.pagination.total_pages) {
      fetchLineData(lineData.pagination.page + 1);
    }
  };

  // Определяем иконку и цвет статуса (аналогично App.tsx)
  let StatusIcon = IconQuestion;
  let statusColorClass = 'text-gray-500';
  if (lineData?.status === 'OK') {
      StatusIcon = IconGear;
      statusColorClass = 'text-green-600';
  } else if (lineData?.status?.startsWith('Ошибка')) { // Добавил ?. для безопасности
      StatusIcon = IconQuestion;
      statusColorClass = 'text-red-600';
  }

  // Получаем заголовки таблицы данных
  const dataColumns = lineData && lineData.data.length > 0 ? Object.keys(lineData.data[0]) : [];

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      {/* Шапка с навигацией назад */}
      <header className="sticky top-0 z-20 border-b bg-white shadow-sm">
        <div className="container flex items-center justify-start h-16 px-4 mx-auto md:px-6">
           <Link
             to="/"
             className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-indigo-500"
           >
              <IconArrowLeft />
              Назад к списку
           </Link>
        </div>
      </header>

      <main className="container px-4 py-8 mx-auto md:px-6">
        {isLoading && (
          <div className="text-center py-10">
             <p className="text-gray-500">Загрузка данных линии...</p>
             {/* Можно добавить спиннер */}
          </div>
        )}

        {error && (
          <div className="relative w-full p-4 mb-6 text-sm text-red-700 bg-red-100 border border-red-300 rounded-md" role="alert">
            <strong className="font-semibold">Ошибка: </strong>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="absolute top-2.5 right-2.5 p-1.5 rounded-md text-red-500 hover:text-red-700 hover:bg-red-200 transition-colors" aria-label="Закрыть">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
        )}

        {!isLoading && !error && lineData && (
          <div className="bg-white border rounded-lg shadow-sm overflow-hidden">
            {/* Информация о линии */}
            <div className="p-5 border-b">
               <h1 className="text-2xl font-bold mb-2">{lineData.area_name}</h1>
               <div className={`flex items-center text-base ${statusColorClass}`}>
                  <StatusIcon />
                  <span>{lineData.status}</span>
               </div>
               {lineData.last_update && (
                  <p className="text-sm text-gray-500 mt-1.5">
                      Последнее обновление: {new Date(lineData.last_update).toLocaleString()}
                  </p>
               )}
               {/* Ссылка на страницу аналитики */} 
               <div className="mt-4">
                  <Link
                      to={`/area/${areaName}/analyse`}
                      className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-purple-700 bg-purple-100 rounded-md hover:bg-purple-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-purple-500"
                  >
                     <IconChartBarSquare />
                     Посмотреть аналитику
                  </Link>
               </div>
            </div>

            {/* Таблица данных */}
            <div className="p-5">
               <h2 className="text-lg font-semibold mb-3">Измерения</h2>
               {lineData.data.length > 0 ? (
                 <div className="overflow-x-auto border rounded-md mb-4">
                   <table className="min-w-full text-sm divide-y divide-gray-200">
                     <thead className="bg-gray-50">
                       <tr>
                         {dataColumns.map((header) => (
                           <th key={header} className="px-4 py-2 font-medium text-left text-gray-500 uppercase tracking-wider whitespace-nowrap">
                             {header}
                           </th>
                         ))}
                       </tr>
                     </thead>
                     <tbody className="divide-y divide-gray-200 bg-white">
                       {lineData.data.map((record, index) => (
                         <tr key={index} className="hover:bg-gray-50">
                           {dataColumns.map((header) => (
                             <td key={header} className="px-4 py-2 whitespace-nowrap text-gray-800">
                               {String(record[header])}
                             </td>
                           ))}
                         </tr>
                       ))}
                     </tbody>
                   </table>
                 </div>
               ) : (
                 <p className="text-sm text-gray-500">Нет данных измерений для отображения.</p>
               )}

                {/* Элементы управления пагинацией */}
                {lineData && lineData.pagination && lineData.pagination.total_pages > 1 && (
                    <div className="flex items-center justify-between text-sm text-gray-600 mt-4">
                        {/* Информация о страницах */}
                        <span>
                            Страница {lineData.pagination.page} из {lineData.pagination.total_pages}
                            (Всего: {lineData.pagination.total_items} записей)
                        </span>
                        {/* Кнопки навигации */}
                        <div className="space-x-2">
                            <button
                                onClick={handlePrevPage}
                                disabled={lineData.pagination.page <= 1 || isLoading}
                                className="px-3 py-1 border rounded-md bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Назад
                            </button>
                            <button
                                onClick={handleNextPage}
                                disabled={lineData.pagination.page >= lineData.pagination.total_pages || isLoading}
                                className="px-3 py-1 border rounded-md bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Вперед
                            </button>
                        </div>
                    </div>
                )}
            </div>
          </div>
        )}

        {!isLoading && !error && !lineData && (
            <p className="text-center text-gray-500">Данные для линии не найдены.</p>
        )}
      </main>
    </div>
  );
}

export default LineDetailPage; 