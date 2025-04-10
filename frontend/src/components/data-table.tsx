import { useState, useMemo } from 'react';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  useReactTable,
  ColumnFiltersState,
  getFilteredRowModel,
  SortingState,
  getSortedRowModel,
  VisibilityState,
} from '@tanstack/react-table';
import { ChevronDown, ChevronLeft, ChevronRight, SlidersHorizontal, BarChart } from 'lucide-react';

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  totalRecords: number;
  onPageChange: (page: number) => void;
  onViewAnalysis?: () => void;
  currentPage: number;
  pageSize: number;
  tableName: string;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
}

export function DataTable<TData, TValue>({
  columns,
  data,
  totalRecords,
  onPageChange,
  onViewAnalysis,
  currentPage,
  pageSize,
  tableName,
  onColumnVisibilityChange,
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [isColumnSelectorOpen, setIsColumnSelectorOpen] = useState(false);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: (updatedVisibility) => {
      setColumnVisibility(updatedVisibility);
      onColumnVisibilityChange?.(updatedVisibility);
    },
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    manualPagination: true,
    pageCount: Math.ceil(totalRecords / pageSize),
  });

  const totalPages = useMemo(() => Math.ceil(totalRecords / pageSize), [totalRecords, pageSize]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">{tableName}</h2>
        <div className="flex items-center gap-2">
          {onViewAnalysis && (
            <button
              onClick={onViewAnalysis}
              className="flex items-center gap-1 px-3 py-1 text-sm text-white bg-blue-600 rounded-md hover:bg-blue-700"
            >
              <BarChart size={16} />
              <span>View Analysis</span>
            </button>
          )}
          <button 
            className="flex items-center gap-1 px-3 py-1 text-sm border rounded-md hover:bg-accent"
            onClick={() => setIsColumnSelectorOpen(!isColumnSelectorOpen)}
          >
            <SlidersHorizontal size={16} />
            <span>Columns</span>
          </button>
        </div>
      </div>

      {isColumnSelectorOpen && (
        <div className="p-4 border rounded-lg bg-card">
          <h3 className="mb-2 text-sm font-medium">Toggle Columns</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {table.getAllColumns()
              .filter(column => column.getCanHide())
              .map(column => (
                <div key={column.id} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id={`column-${column.id}`}
                    checked={column.getIsVisible()}
                    onChange={column.getToggleVisibilityHandler()}
                    className="rounded border-input"
                  />
                  <label htmlFor={`column-${column.id}`} className="text-sm">
                    {column.id}
                  </label>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="border rounded-md">
        <div className="overflow-x-auto">
          <div className="max-h-[400px] overflow-y-auto">
            <table className="w-full">
              <thead className="border-b bg-muted/50 sticky top-0 z-10">
                {table.getHeaderGroups().map(headerGroup => (
                  <tr key={headerGroup.id}>
                    {headerGroup.headers.map(header => (
                      <th key={header.id} className="px-4 py-3 text-sm font-medium text-left">
                        {header.isPlaceholder ? null : (
                          <div
                            className={`flex items-center gap-1 ${
                              header.column.getCanSort() ? 'cursor-pointer select-none' : ''
                            }`}
                            onClick={header.column.getToggleSortingHandler()}
                          >
                            {flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                            {header.column.getCanSort() && (
                              <ChevronDown
                                size={16}
                                className={`transition-transform ${
                                  header.column.getIsSorted() === 'asc'
                                    ? 'rotate-180'
                                    : header.column.getIsSorted() === 'desc'
                                    ? 'rotate-0'
                                    : 'rotate-0 opacity-0'
                                }`}
                              />
                            )}
                          </div>
                        )}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.length ? (
                  table.getRowModel().rows.map(row => (
                    <tr key={row.id} className="border-b hover:bg-muted/50">
                      {row.getVisibleCells().map(cell => (
                        <td key={cell.id} className="px-4 py-3 text-sm">
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={columns.length} className="py-6 text-center">
                      No results found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="flex items-center justify-between px-4 py-2 border-t">
          <div className="text-sm text-muted-foreground">
            {totalRecords > 0 
              ? `Showing ${(currentPage - 1) * pageSize + 1} to ${Math.min(currentPage * pageSize, totalRecords)} of ${totalRecords} entries`
              : 'No entries'
            }
          </div>
          <div className="flex items-center gap-2">
            <button
              className="p-1 border rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm">
              Page {currentPage} of {totalPages || 1}
            </span>
            <button
              className="p-1 border rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
} 