import { useState, useEffect, useMemo } from 'react'
import { useTheme } from './context/theme-provider'
import { ThemeToggle } from './components/theme-toggle'
import { DataTable } from './components/data-table'
import { fetchTables, fetchTableData, fetchAnalysis, TableInfo, TableData, AnalysisResult } from './lib/api'
import { ColumnDef } from '@tanstack/react-table'

interface AnalysisModalProps {
  isOpen: boolean
  onClose: () => void
  analysisData: AnalysisResult | null
}

function AnalysisModal({ isOpen, onClose, analysisData }: AnalysisModalProps) {
  if (!isOpen || !analysisData) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-3xl max-h-[80vh] overflow-auto p-6 rounded-lg bg-card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">
            Analysis Results: {analysisData.table_name}
            {analysisData.production_line && ` (Line ${analysisData.production_line})`}
          </h2>
          <button
            onClick={onClose}
            className="p-2 text-muted-foreground hover:text-foreground"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          <div className="p-4 border rounded-lg">
            <h3 className="mb-2 font-medium">Average Values</h3>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {Object.entries(analysisData.summary.average_values).map(([key, value]) => (
                <div key={key} className="p-3 border rounded-md">
                  <div className="text-sm text-muted-foreground">{key}</div>
                  <div className="text-xl font-semibold">{value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="mb-2 font-medium">Trends</h3>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {Object.entries(analysisData.summary.trends).map(([key, value]) => (
                <div key={key} className="p-3 border rounded-md">
                  <div className="text-sm text-muted-foreground">{key}</div>
                  <div className="font-medium capitalize">{value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="mb-2 font-medium">Anomalies Detected</h3>
            <div className="p-3 border rounded-md">
              <div className="text-xl font-semibold">{analysisData.summary.anomalies_detected}</div>
            </div>
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="mb-2 font-medium">Recommendations</h3>
            <ul className="pl-5 space-y-1 list-disc">
              {analysisData.recommendations.map((recommendation, i) => (
                <li key={i}>{recommendation}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="flex justify-end mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 border rounded-md hover:bg-accent"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function App() {
  const { theme } = useTheme()
  const [tables, setTables] = useState<TableInfo[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [tableData, setTableData] = useState<TableData | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(50)
  const [isLoading, setIsLoading] = useState(false)
  const [visibleColumns, setVisibleColumns] = useState<Record<string, boolean>>({})
  const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false)
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshTimestamp, setRefreshTimestamp] = useState<Date>(new Date())

  // Auto-refresh every 10 minutes
  useEffect(() => {
    const intervalId = setInterval(() => {
      setRefreshTimestamp(new Date());
    }, 600000); // 10 minutes = 600000 ms
    
    return () => clearInterval(intervalId);
  }, []);

  // Load tables on initial render
  useEffect(() => {
    const loadTables = async () => {
      try {
        setIsLoading(true)
        const tablesData = await fetchTables()
        setTables(tablesData)
        if (tablesData.length > 0 && !selectedTable) {
          setSelectedTable(tablesData[0].name)
        }
        setError(null)
      } catch (err) {
        console.error('Failed to load tables:', err)
        setError('Failed to load database tables. Please try again later.')
      } finally {
        setIsLoading(false)
      }
    }

    loadTables()
  }, [])

  // Load table data when selected table changes or on refresh
  useEffect(() => {
    if (!selectedTable) return

    const loadTableData = async () => {
      try {
        setIsLoading(true)
        const data = await fetchTableData(
          selectedTable,
          currentPage,
          pageSize,
          Object.entries(visibleColumns)
            .filter(([_, isVisible]) => isVisible)
            .map(([column]) => column)
        )
        setTableData(data)
        setError(null)
      } catch (err) {
        console.error('Failed to load table data:', err)
        setError('Failed to load table data. Please try again later.')
        setTableData(null)
      } finally {
        setIsLoading(false)
      }
    }

    loadTableData()
  }, [selectedTable, currentPage, pageSize, visibleColumns, refreshTimestamp])

  // Initialize visible columns when table data changes
  useEffect(() => {
    if (tableData) {
      const initialVisibility: Record<string, boolean> = {}
      tableData.columns.forEach(column => {
        initialVisibility[column] = true
      })
      setVisibleColumns(initialVisibility)
    }
  }, [tableData?.table_name])

  const handleViewAnalysis = async () => {
    try {
      setIsLoading(true)
      let productionLine: number | undefined = undefined
      
      // Если в названии таблицы есть номер линии, попробуем его извлечь
      const lineMatch = selectedTable.match(/production_line_(\d+)/);
      if (lineMatch && lineMatch[1]) {
        productionLine = Number(lineMatch[1]);
      }
      
      const analysisResult = await fetchAnalysis(selectedTable, productionLine)
      setAnalysisData(analysisResult)
      setIsAnalysisModalOpen(true)
      setError(null)
    } catch (err) {
      console.error('Failed to load analysis:', err)
      setError('Failed to load analysis data. Please try again later.')
    } finally {
      setIsLoading(false)
    }
  }

  // Create dynamic columns for the data table
  const columns = useMemo<ColumnDef<any, any>[]>(() => {
    if (!tableData) return []

    return tableData.columns.map(col => ({
      accessorKey: col,
      header: col,
      cell: ({ getValue }) => {
        const value = getValue()
        // Handle different value types
        if (value === null || value === undefined) return '-'
        if (typeof value === 'boolean') return value ? 'Yes' : 'No'
        if (value instanceof Date) return value.toLocaleString()
        return value.toString()
      },
    }))
  }, [tableData])

  return (
    <div className={`min-h-screen bg-background text-foreground`}>
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="container flex items-center justify-between h-16 mx-auto">
          <h1 className="text-xl font-bold">Production Monitoring Dashboard</h1>
          <ThemeToggle />
        </div>
      </header>

      <main className="container py-6 mx-auto">
        {error && (
          <div className="p-4 mb-6 text-white bg-red-500 rounded-md">
            {error}
          </div>
        )}

        <div className="mb-6 space-y-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="w-full sm:w-64">
              <label htmlFor="table-select" className="block mb-1 text-sm">
                Select Production Line Data
              </label>
              <select
                id="table-select"
                className="w-full p-2 border rounded-md bg-background"
                value={selectedTable}
                onChange={(e) => {
                  setSelectedTable(e.target.value)
                  setCurrentPage(1)
                }}
                disabled={isLoading || tables.length === 0}
              >
                {tables.length === 0 ? (
                  <option value="">No tables available</option>
                ) : (
                  tables.map((table) => (
                    <option key={table.name} value={table.name}>
                      {table.name}
                    </option>
                  ))
                )}
              </select>
            </div>
            <div className="text-sm text-muted-foreground">
              Last updated: {refreshTimestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>

        {isLoading && !tableData ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-lg">Loading data...</div>
          </div>
        ) : tableData ? (
          <DataTable
            columns={columns}
            data={tableData.data}
            totalRecords={tableData.total_records}
            onPageChange={setCurrentPage}
            currentPage={currentPage}
            pageSize={pageSize}
            tableName={tableData.table_name}
            onViewAnalysis={handleViewAnalysis}
            onColumnVisibilityChange={(visibility) => {
              const newVisibleColumns: Record<string, boolean> = {}
              Object.keys(visibility).forEach((key) => {
                newVisibleColumns[key] = visibility[key]
              })
              setVisibleColumns(newVisibleColumns)
            }}
          />
        ) : (
          <div className="flex items-center justify-center h-64">
            <div className="text-lg">Select a table to view data</div>
          </div>
        )}
      </main>

      <AnalysisModal
        isOpen={isAnalysisModalOpen}
        onClose={() => setIsAnalysisModalOpen(false)}
        analysisData={analysisData}
      />
    </div>
  )
}

export default App 