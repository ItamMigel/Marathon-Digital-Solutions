import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface TableInfo {
  name: string;
  columns: string[];
}

export interface TableData {
  table_name: string;
  columns: string[];
  data: Record<string, any>[];
  total_records: number;
}

export interface AnalysisResult {
  table_name: string;
  production_line?: number;
  summary: Record<string, any>;
  recommendations: string[];
}

export const fetchTables = async (): Promise<TableInfo[]> => {
  const response = await api.get<TableInfo[]>('/tables');
  return response.data;
};

export const fetchTableData = async (
  tableName: string,
  page: number = 1,
  pageSize: number = 50,
  selectedColumns?: string[],
  filters?: Record<string, string>
): Promise<TableData> => {
  const params: Record<string, any> = {
    page,
    page_size: pageSize,
  };

  if (selectedColumns && selectedColumns.length > 0) {
    params.columns = selectedColumns.join(',');
  }

  if (filters) {
    const filterParts: string[] = [];
    Object.entries(filters).forEach(([key, value]) => {
      if (value) {
        filterParts.push(`${key}:eq:${value}`);
      }
    });
    if (filterParts.length > 0) {
      params.filters = filterParts.join(',');
    }
  }

  const response = await api.get<TableData>(`/tables/${tableName}`, { params });
  return response.data;
};

export const fetchAnalysis = async (
  tableName: string,
  productionLine?: number
): Promise<AnalysisResult> => {
  const params: Record<string, any> = {};
  
  if (productionLine !== undefined) {
    params.production_line = productionLine;
  }
  
  const response = await api.get<AnalysisResult>(`/analysis/${tableName}`, { params });
  return response.data;
}; 