import { useTheme } from './ThemeProvider'
import { Button } from '@/components/ui/button'
import { Sun, Moon, Bell, RefreshCw } from 'lucide-react'
import { useState, useEffect } from 'react'
import { getHealth } from '@/lib/api'
import type { HealthResponse } from '@/lib/types'

interface HeaderProps {
    title: string
    description?: string
}

export function Header({ title, description }: HeaderProps) {
    const { theme, setTheme, resolvedTheme } = useTheme()
    const [health, setHealth] = useState<HealthResponse | null>(null)
    const [loading, setLoading] = useState(false)

    const fetchHealth = async () => {
        setLoading(true)
        try {
            const data = await getHealth()
            setHealth(data)
        } catch (error) {
            console.error('Health check failed:', error)
            setHealth(null)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchHealth()
        const interval = setInterval(fetchHealth, 30000) // Every 30s
        return () => clearInterval(interval)
    }, [])

    const toggleTheme = () => {
        setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')
    }

    return (
        <header className="h-16 border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-30">
            <div className="h-full px-6 flex items-center justify-between">
                {/* Title */}
                <div>
                    <h1 className="text-xl font-semibold">{title}</h1>
                    {description && (
                        <p className="text-sm text-muted-foreground">{description}</p>
                    )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-4">
                    {/* API Status */}
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted">
                        <div
                            className={`status-dot ${health?.status === 'healthy' ? 'online' : 'offline'
                                }`}
                        />
                        <span className="text-sm">
                            {health?.status === 'healthy' ? 'Connected' : 'Disconnected'}
                        </span>
                        <button
                            onClick={fetchHealth}
                            disabled={loading}
                            className="p-1 hover:bg-background rounded transition-colors"
                        >
                            <RefreshCw
                                className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`}
                            />
                        </button>
                    </div>

                    {/* GPU Status */}
                    {health?.gpu_available && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 text-green-500 border border-green-500/20">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-sm font-medium">
                                {health.gpu_name?.split(' ')[0] || 'GPU'}
                            </span>
                        </div>
                    )}

                    {/* Notifications */}
                    <Button variant="ghost" size="icon" className="relative">
                        <Bell className="w-5 h-5" />
                        <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-severity-critical text-white text-[10px] rounded-full flex items-center justify-center">
                            3
                        </span>
                    </Button>

                    {/* Theme Toggle */}
                    <Button variant="ghost" size="icon" onClick={toggleTheme}>
                        {resolvedTheme === 'dark' ? (
                            <Sun className="w-5 h-5" />
                        ) : (
                            <Moon className="w-5 h-5" />
                        )}
                    </Button>
                </div>
            </div>
        </header>
    )
}
