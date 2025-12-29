import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getDefectSummary, getDefectTrends } from '@/lib/api'
import { formatNumber, formatPercent, getSeverityColor } from '@/lib/utils'
import type { DefectSummary, TrendsResponse } from '@/lib/types'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    PieChart,
    Pie,
    Cell,
} from 'recharts'
import {
    AlertTriangle,
    CheckCircle,
    Activity,
    TrendingUp,
    Clock,
    Cpu,
} from 'lucide-react'

export default function Dashboard() {
    const [summary, setSummary] = useState<DefectSummary | null>(null)
    const [trends, setTrends] = useState<TrendsResponse | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [summaryData, trendsData] = await Promise.all([
                    getDefectSummary(24),
                    getDefectTrends('hour', 24),
                ])
                setSummary(summaryData)
                setTrends(trendsData)
            } catch (error) {
                console.error('Failed to fetch dashboard data:', error)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
            </div>
        )
    }

    const kpis = [
        {
            title: 'Total Inspections',
            value: formatNumber(summary?.total_inspections || 0),
            icon: Activity,
            change: '+12%',
            color: 'text-blue-500',
        },
        {
            title: 'Defects Detected',
            value: formatNumber(summary?.total_defects || 0),
            icon: AlertTriangle,
            change: summary?.detection_rate ? formatPercent(summary.detection_rate) : '0%',
            color: 'text-orange-500',
        },
        {
            title: 'Pass Rate',
            value: formatPercent(1 - (summary?.detection_rate || 0)),
            icon: CheckCircle,
            change: '+5.2%',
            color: 'text-green-500',
            positive: true,
        },
        {
            title: 'Avg Processing',
            value: '45ms',
            icon: Clock,
            change: '-15%',
            color: 'text-purple-500',
            positive: true,
        },
    ]

    const severityData = summary?.defects_by_severity
        ? Object.entries(summary.defects_by_severity).map(([name, value]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            value,
            color: getSeverityColor(name),
        }))
        : []

    const typeData = summary?.defects_by_type
        ? Object.entries(summary.defects_by_type).map(([name, value]) => ({
            name: name.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
            value,
        }))
        : []

    return (
        <main className="p-6 space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {kpis.map((kpi, idx) => (
                    <Card key={idx} className="kpi-card">
                        <CardContent className="p-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-muted-foreground">{kpi.title}</p>
                                    <p className="text-3xl font-bold mt-2">{kpi.value}</p>
                                    <Badge
                                        variant={kpi.positive ? 'low' : 'secondary'}
                                        className="mt-2"
                                    >
                                        <TrendingUp className="w-3 h-3 mr-1" />
                                        {kpi.change}
                                    </Badge>
                                </div>
                                <div className={`p-3 rounded-full bg-muted ${kpi.color}`}>
                                    <kpi.icon className="w-6 h-6" />
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Defect Trend Chart */}
                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Activity className="w-5 h-5" />
                            Defect Trend (24h)
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={trends?.data || []}
                                    margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                                    <XAxis
                                        dataKey="period"
                                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                                        tickFormatter={(v) => v.split(' ')[1] || v}
                                    />
                                    <YAxis tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'hsl(var(--card))',
                                            border: '1px solid hsl(var(--border))',
                                            borderRadius: '8px',
                                        }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="total"
                                        stroke="hsl(var(--primary))"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 6 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* Severity Distribution */}
                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <AlertTriangle className="w-5 h-5" />
                            Severity Distribution
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[300px] flex items-center">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={severityData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={100}
                                        paddingAngle={4}
                                        dataKey="value"
                                        label={({ name, percent }) =>
                                            `${name} ${(percent * 100).toFixed(0)}%`
                                        }
                                        labelLine={false}
                                    >
                                        {severityData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Defect Types Bar Chart */}
            <Card className="glass-card">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Cpu className="w-5 h-5" />
                        Defects by Type
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={typeData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                                <XAxis
                                    dataKey="name"
                                    tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                                    angle={-45}
                                    textAnchor="end"
                                    height={80}
                                />
                                <YAxis tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'hsl(var(--card))',
                                        border: '1px solid hsl(var(--border))',
                                        borderRadius: '8px',
                                    }}
                                />
                                <Bar
                                    dataKey="value"
                                    fill="hsl(var(--primary))"
                                    radius={[4, 4, 0, 0]}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </main>
    )
}
