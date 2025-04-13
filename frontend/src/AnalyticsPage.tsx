import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// --- Иконки (можно вынести) ---
const IconArrowLeft = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-1">
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
  </svg>
);

const IconChartBar = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-1.5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
  </svg>
);


// --- Интерфейсы (должны совпадать с Pydantic моделями на бэкенде) ---
interface TrendDataPoint {
  Время: string; // Получаем как строку ISO
  value: number | null;
}

interface ColumnAnalytics {
  count: number;
  min: number | null;
  max: number | null;
  average: number | null;
  median: number | null;
  std_dev: number | null;
}

interface AnalyticsResponse {
  area_name: string;
  start_time: string;
  end_time: string;
  requested_minutes: number;
  statistics: { [key: string]: ColumnAnalytics };
  trends: { [key: string]: TrendDataPoint[] };
}

const API_BASE_URL = 'http://localhost:8000';

// --- Доступные периоды анализа (в минутах) ---
const ANALYSIS_PERIODS = [
  { label: '1 час', value: 60 },
  { label: '6 часов', value: 360 },
  { label: '24 часа', value: 1440 },
];

function AnalyticsPage() {
  const { areaName } = useParams<{ areaName: string }>();
  const [analyticsData, setAnalyticsData] = useState<AnalyticsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<number>(ANALYSIS_PERIODS[0].value); // По умолчанию 60 минут
  const [selectedTrendColumns, setSelectedTrendColumns] = useState<string[]>([]); // Выбранные колонки для графиков

  const fetchAnalyticsData = useCallback(async (minutes: number) => { // Принимаем минуты как аргумент
    if (!areaName) return;
    setIsLoading(true);
    setError(null);
    // Сбрасываем выбранные графики при смене периода, так как данные могут измениться
    setSelectedTrendColumns([]); 
    try {
      const response = await axios.get<AnalyticsResponse>(
        `${API_BASE_URL}/api/lines/${areaName}/analytics`,
        { params: { minutes } } // Используем переданные минуты
      );
      setAnalyticsData(response.data);
      // По умолчанию выбираем первые 3 доступных тренда, если есть
      if (response.data.trends) {
        const initialTrends = Object.keys(response.data.trends).slice(0, 3);
        setSelectedTrendColumns(initialTrends);
      }
    } catch (err: any) {
      console.error(`Ошибка загрузки аналитики для ${areaName} за ${minutes} минут:`, err);
      setError(err.response?.data?.detail || err.message || 'Не удалось загрузить данные аналитики.');
      setAnalyticsData(null);
    } finally {
      setIsLoading(false);
    }
  }, [areaName]); // Убираем minutes из зависимостей, чтобы не вызывать лишний раз при смене периода

  useEffect(() => {
    fetchAnalyticsData(selectedPeriod); // Вызываем с текущим выбранным периодом
  }, [fetchAnalyticsData, selectedPeriod]); // Добавляем selectedPeriod в зависимости

  // Обработчик выбора периода
  const handlePeriodChange = (minutes: number) => {
    setSelectedPeriod(minutes);
    // Перезагрузка данных с новым периодом произойдет через useEffect
  };

  // Обработчик выбора/снятия колонки для графика
  const handleTrendSelectionChange = (columnName: string) => {
    setSelectedTrendColumns(prevSelected =>
      prevSelected.includes(columnName)
        ? prevSelected.filter(col => col !== columnName) // Снять выбор
        : [...prevSelected, columnName] // Добавить в выбор
    );
  };

  // Функция для форматирования данных для графика
  const formatChartData = (trendData: TrendDataPoint[]) => {
    return trendData
      .map(point => ({
        time: new Date(point.Время).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        value: point.value,
      }))
      .filter(point => point.value !== null);
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      {/* Шапка */}
      <header className="sticky top-0 z-20 border-b bg-white shadow-sm">
        <div className="container flex items-center justify-between h-16 px-4 mx-auto md:px-6">
          <div className="flex items-center">
            <Link
              to={`/area/${areaName}`}
              className="inline-flex items-center mr-4 px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-indigo-500"
            >
              <IconArrowLeft />
              Назад к деталям
            </Link>
            <h1 className="text-xl font-semibold">Аналитика: {areaName}</h1>
          </div>
          {/* === Выбор периода === */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Период:</span>
            {ANALYSIS_PERIODS.map(period => (
              <button
                key={period.value}
                onClick={() => handlePeriodChange(period.value)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${selectedPeriod === period.value
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
              >
                {period.label}
              </button>
            ))}
          </div>
          {/* Можно добавить кнопку обновления */}
        </div>
      </header>

      <main className="container px-4 py-8 mx-auto md:px-6">
        {isLoading && (
          <div className="text-center py-10">
            <p className="text-gray-500">Загрузка данных аналитики...</p>
          </div>
        )}

        {error && (
          <div className="relative w-full p-4 mb-6 text-sm text-red-700 bg-red-100 border border-red-300 rounded-md" role="alert">
            <strong className="font-semibold">Ошибка: </strong>
            <span>{error}</span>
            {/* Кнопка закрытия ошибки */} 
          </div>
        )}

        {!isLoading && !error && analyticsData && (
          <div className="space-y-8">
            {/* Информация о периоде */}
            <div className="p-4 bg-white border rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold mb-2">Анализ за период:</h2>
              <p className="text-sm text-gray-600">
                С {new Date(analyticsData.start_time).toLocaleString()} по {new Date(analyticsData.end_time).toLocaleString()} ({analyticsData.requested_minutes} мин.)
              </p>
            </div>

            {/* Статистика по колонкам */}
            <div className="p-4 bg-white border rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold mb-3">Статистика по параметрам (Выберите для графика):</h2>
              {Object.keys(analyticsData.statistics).length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {Object.entries(analyticsData.statistics).map(([colName, stats]) => (
                    <div
                      key={colName}
                      className={`relative p-3 border rounded transition-colors duration-150 ${analyticsData.trends && colName in analyticsData.trends ? 'cursor-pointer hover:bg-indigo-50' : 'bg-gray-50 opacity-70'} ${selectedTrendColumns.includes(colName) ? 'border-indigo-400 ring-2 ring-indigo-200 bg-indigo-50' : 'border-gray-200'}`}
                      onClick={() => analyticsData.trends && colName in analyticsData.trends && handleTrendSelectionChange(colName)}
                      title={analyticsData.trends && colName in analyticsData.trends ? "Нажмите, чтобы добавить/убрать график" : "График для этого параметра недоступен"}
                    >
                     {analyticsData.trends && colName in analyticsData.trends && (
                        <input
                            type="checkbox"
                            checked={selectedTrendColumns.includes(colName)}
                            readOnly
                            className="absolute top-2 right-2 h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500 cursor-pointer"
                            // Клик обрабатывается на родительском div
                        />
                     )}
                      <h3 className={`text-sm font-medium mb-2 ${analyticsData.trends && colName in analyticsData.trends ? 'text-gray-800' : 'text-gray-500'}`}>{colName.replace(/_/g, ' ')}</h3>
                      {stats.count > 0 ? (
                        <dl className="text-xs space-y-1 text-gray-600">
                          <div className="flex justify-between"><dt>Точек:</dt><dd className="font-mono">{stats.count}</dd></div>
                          <div className="flex justify-between"><dt>Мин:</dt><dd className="font-mono">{stats.min?.toFixed(2) ?? '-'}</dd></div>
                          <div className="flex justify-between"><dt>Макс:</dt><dd className="font-mono">{stats.max?.toFixed(2) ?? '-'}</dd></div>
                          <div className="flex justify-between"><dt>Среднее:</dt><dd className="font-mono">{stats.average?.toFixed(2) ?? '-'}</dd></div>
                          <div className="flex justify-between"><dt>Медиана:</dt><dd className="font-mono">{stats.median?.toFixed(2) ?? '-'}</dd></div>
                          <div className="flex justify-between"><dt>Стд. откл.:</dt><dd className="font-mono">{stats.std_dev?.toFixed(2) ?? '-'}</dd></div>
                        </dl>
                      ) : (
                        <p className="text-xs text-gray-400 italic">Нет данных для расчета</p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">Статистика не рассчитана.</p>
              )}
            </div>

            {/* Графики трендов */}
            <div className="p-4 bg-white border rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold mb-4">Графики трендов:</h2>
              {selectedTrendColumns.length > 0 && analyticsData.trends ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {selectedTrendColumns.map((colName) => {
                    // Убедимся, что данные для этого тренда есть
                    const trendData = analyticsData.trends[colName];
                    if (!trendData) return null; // Пропустить, если данных нет

                    const chartData = formatChartData(trendData);
                    return (
                      <div key={colName} className="h-64"> {/* Задаем высоту контейнера */}
                        <h3 className="text-md font-medium mb-2 text-center text-gray-700">{colName.replace(/_/g, ' ')}</h3>
                        {chartData.length > 1 ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false}/>
                              <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd"/>
                              <YAxis tick={{ fontSize: 10 }} domain={['auto', 'auto']}/>
                              <Tooltip
                                contentStyle={{ fontSize: 12, padding: '5px 10px' }}
                                labelFormatter={(label) => `Время: ${label}`}
                                formatter={(value: number, name: string) => [value.toFixed(2), name]}
                              />
                              <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={1.5} dot={false} name={colName.replace(/_/g, ' ')}/>
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="flex items-center justify-center h-full text-xs text-gray-400">Мало данных для графика</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-gray-500">
                  {analyticsData.trends && Object.keys(analyticsData.trends).length > 0
                    ? "Выберите параметры из статистики выше для построения графиков."
                    : "Нет данных для построения трендов."
                  }
                </p>
              )}
            </div>
          </div>
        )}

        {!isLoading && !error && !analyticsData && (
          <p className="text-center text-gray-500">Не удалось загрузить данные аналитики.</p>
        )}
      </main>
    </div>
  );
}

export default AnalyticsPage; 