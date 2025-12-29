import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { listDefects, getDefectTrends } from '@/lib/api'
import { formatDateTime, formatPercent } from '@/lib/utils'
import type { DefectListResponse, TrendsResponse } from '@/lib/types'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
} from 'recharts'
import {
    ChevronLeft,
    ChevronRight,
    Filter,
    Download,
    BarChart3,
    Table as TableIcon,
} from 'lucide-react'

export default function Analytics() {
    const [data, setData] = useState<DefectListResponse | null>(null)
    const [trends, setTrends] = useState<TrendsResponse | null>(null)
    const [page, setPage] = useState(1)
    const [loading, setLoading] = useState(true)
    const [trendPeriod, setTrendPeriod] = useState<'hour' | 'day' | 'week'>('day')

    useEffect(() => {
        fetchData()
    }, [page])

    useEffect(() => {
        fetchTrends()
    }, [trendPeriod])

    const fetchData = async () => {
        setLoading(true)
        try {
            const result = await listDefects({ page, pageSize: 15 })
            setData(result)
        } catch (error) {
            console.error('Failed to fetch defects:', error)
        } finally {
            setLoading(false)
        }
    }

    const fetchTrends = async () => {
        try {
            const result = await getDefectTrends(trendPeriod, trendPeriod === 'hour' ? 48 : 30)
            setTrends(result)
        } catch (error) {
            console.error('Failed to fetch trends:', error)
        }
    }

    const severityColors: Record<string, string> = {
        critical: '#ef4444',
        high: '#f97316',
        medium: '#eab308',
        low: '#22c55e',
        info: '#3b82f6',
    }

    return (
        <main className="p-6 space-y-6">
            {/* Trends Section */}
            <Card className="glass-card">
                <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="w-5 h-5" />
                        Defect Trends
                    </CardTitle>
                    <Tabs value={trendPeriod} onValueChange={(v) => setTrendPeriod(v as typeof trendPeriod)}>
                        <TabsList>
                            <TabsTrigger value="hour">Hourly</TabsTrigger>
                            <TabsTrigger value="day">Daily</TabsTrigger>
                            <TabsTrigger value="week">Weekly</TabsTrigger>
                        </TabsList>
                    </Tabs>
                </CardHeader>
                <CardContent>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={trends?.data || []}>
                                <defs>
                                    <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                                <XAxis
                                    dataKey="period"
                                    tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                                    tickFormatter={(v) => {
                                        if (trendPeriod === 'hour') return v.split(' ')[1] || v
                                        return v.split('-').slice(1).join('/')
                                    }}
                                />
                                <YAxis tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'hsl(var(--card))',
                                        border: '1px solid hsl(var(--border))',
                                        borderRadius: '8px',
                                    }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="total"
                                    stroke="hsl(var(--primary))"
                                    fillOpacity={1}
                                    fill="url(#colorTotal)"
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>

            {/* Data Table */}
            <Card className="glass-card">
                <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                        <TableIcon className="w-5 h-5" />
                        Defect Records
                    </CardTitle>
                    <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm">
                            <Filter className="w-4 h-4 mr-2" />
                            Filter
                        </Button>
                        <Button variant="outline" size="sm">
                            <Download className="w-4 h-4 mr-2" />
                            Export
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    {loading ? (
                        <div className="flex items-center justify-center h-48">
                            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
                        </div>
                    ) : (
                        <>
                            <div className="overflow-x-auto">
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>ID</th>
                                            <th>Type</th>
                                            <th>Severity</th>
                                            <th>Confidence</th>
                                            <th>Area %</th>
                                            <th>Detected</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {data?.defects.map((defect) => (
                                            <tr key={defect.id}>
                                                <td className="font-mono text-xs">{defect.id}</td>
                                                <td className="capitalize">{defect.defect_type.replace('_', ' ')}</td>
                                                <td>
                                                    <Badge variant={defect.severity as 'critical' | 'high' | 'medium' | 'low' | 'info'}>
                                                        {defect.severity}
                                                    </Badge>
                                                </td>
                                                <td>{formatPercent(defect.confidence)}</td>
                                                <td>{defect.area_percentage.toFixed(2)}%</td>
                                                <td className="text-muted-foreground">
                                                    {formatDateTime(defect.created_at)}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {/* Pagination */}
                            <div className="flex items-center justify-between mt-4">
                                <p className="text-sm text-muted-foreground">
                                    Page {data?.page} of {data?.total_pages}
                                </p>
                                <div className="flex items-center gap-2">
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setPage(p => Math.max(1, p - 1))}
                                        disabled={page <= 1}
                                    >
                                        <ChevronLeft className="w-4 h-4" />
                                        Previous
                                    </Button>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setPage(p => p + 1)}
                                        disabled={page >= (data?.total_pages || 1)}
                                    >
                                        Next
                                        <ChevronRight className="w-4 h-4" />
                                    </Button>
                                </div>
                            </div>
                        </>
                    )}
                </CardContent>
            </Card>

            {/* Summary Stats */}
            {data?.summary && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <Card className="glass-card p-4">
                        <p className="text-sm text-muted-foreground">Total Defects</p>
                        <p className="text-2xl font-bold">{data.summary.total_defects}</p>
                    </Card>
                    <Card className="glass-card p-4">
                        <p className="text-sm text-muted-foreground">Avg Severity</p>
                        <p className="text-2xl font-bold">{formatPercent(data.summary.avg_severity_score)}</p>
                    </Card>
                    <Card className="glass-card p-4">
                        <p className="text-sm text-muted-foreground">Detection Rate</p>
                        <p className="text-2xl font-bold">{formatPercent(data.summary.detection_rate)}</p>
                    </Card>
                    <Card className="glass-card p-4">
                        <p className="text-sm text-muted-foreground">Inspections</p>
                        <p className="text-2xl font-bold">{data.summary.total_inspections}</p>
                    </Card>
                </div>
            )}
        </main>
    )
}
