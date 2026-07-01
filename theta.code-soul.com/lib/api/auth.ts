/**
 * Authentication API Client
 * 适配 theta_1-main/api 后端认证接口
 *
 * 后端端点：
 *   POST /api/auth/login   → {access_token, token_type, expires_in, user: {username, role, created_at}}
 *   POST /api/auth/logout
 *   GET  /api/auth/me       → {username, role, expires_at}
 *   POST /api/auth/verify   → {valid, username, role, expires_at}
 */

import { apiFetch, API_BASE } from './config';

// ==================== 类型定义 ====================

export interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  created_at: string;
  is_active: boolean;
  role?: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
  user?: User;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface ProfileUpdateRequest {
  email?: string;
  full_name?: string;
}

export interface PasswordChangeRequest {
  current_password: string;
  new_password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  code: string;  // 邮箱验证码
}

export interface SendCodeRequest {
  email: string;
  type: "register" | "reset_password";
}

export interface VerifyCodeRequest {
  email: string;
  code: string;
}

/** 将后端返回的用户/token 信息归一化为前端 User 格式 */
function normalizeUser(raw: any, fallbackUsername = 'unknown'): User {
  return {
    id: raw.id ?? 0,
    username: raw.username ?? fallbackUsername,
    email: raw.email ?? '',
    full_name: raw.full_name ?? raw.username ?? fallbackUsername,
    created_at: raw.created_at ?? raw.expires_at ?? '',
    is_active: raw.is_active ?? true,
    role: raw.role ?? 'user',
  };
}

// ==================== API ====================

export const AuthAPI = {
  /**
   * 注册新用户
   * POST /api/auth/register  body: {username, email, password, full_name?}
   */
  async register(data: RegisterRequest): Promise<User> {
    const raw = await apiFetch<any>(API_BASE, '/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
      timeoutMs: 12_000,
    });
    return normalizeUser(raw);
  },

  /**
   * 登录
   * theta_1-main: POST /api/auth/login  body: {username, password} (form-urlencoded)
   */
  async login(data: LoginRequest): Promise<Token> {
    const form = new URLSearchParams();
    form.set('username', data.username);
    form.set('password', data.password);

    const response = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      body: form.toString(),
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || '用户名或密码错误');
    }

    const raw = await response.json();

    const user: User = raw.user
      ? normalizeUser(raw.user, data.username)
      : normalizeUser({ username: data.username });

    return {
      access_token: raw.access_token,
      token_type: raw.token_type ?? 'bearer',
      expires_in: raw.expires_in ?? 86400,
      user,
    };
  },

  /**
   * 获取当前用户信息
   * 后端没有 /api/auth/me，从 stored user 或 JWT decode 获取
   */
  async getCurrentUser(): Promise<User> {
    // 优先使用 stored user
    const stored = this.getStoredUser();
    if (stored) return stored;

    // 尝试从 JWT token 解码用户信息
    const token = localStorage.getItem('access_token');
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return {
          id: payload.id || payload.sub || 0,
          username: payload.username || payload.sub || 'unknown',
          email: payload.email || '',
          full_name: payload.full_name || '',
          created_at: '',
          is_active: true,
          role: 'user',
        };
      } catch {
        // ignore
      }
    }

    throw new Error('无法获取用户信息');
  },

  /**
   * 验证 Token
   * POST /api/auth/verify  body: token string
   */
  async verifyToken(): Promise<{ valid: boolean; username: string; user_id: number }> {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) return { valid: false, username: '', user_id: 0 };

      const raw = await apiFetch<any>(API_BASE, '/api/auth/verify', {
        method: 'POST',
        body: JSON.stringify(token),
      });
      return {
        valid: raw.valid ?? true,
        username: raw.username ?? '',
        user_id: raw.user_id ?? 0,
      };
    } catch {
      try {
        const user = await this.getCurrentUser();
        return { valid: true, username: user.username, user_id: 0 };
      } catch {
        return { valid: false, username: '', user_id: 0 };
      }
    }
  },

  async logout(): Promise<void> {
    try {
      await apiFetch(API_BASE, '/api/auth/logout', { method: 'POST' });
    } catch {
      // 忽略后端登出失败
    }
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  },

  getToken(): string | null {
    return localStorage.getItem('access_token');
  },

  setAuth(token: string, user: User): void {
    localStorage.setItem('access_token', token);
    localStorage.setItem('user', JSON.stringify(user));
  },

  getStoredUser(): User | null {
    const s = localStorage.getItem('user');
    if (!s) return null;
    try { return JSON.parse(s); } catch { return null; }
  },

  async updateProfile(data: ProfileUpdateRequest): Promise<User> {
    return apiFetch<User>(API_BASE, '/api/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async changePassword(data: PasswordChangeRequest): Promise<{ message: string }> {
    return apiFetch(API_BASE, '/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  /**
   * 发送邮箱验证码
   */
  async sendVerificationCode(data: SendCodeRequest): Promise<{ message: string }> {
    return apiFetch(API_BASE, '/api/auth/send-code', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  /**
   * 验证邮箱验证码
   */
  async verifyCode(data: VerifyCodeRequest): Promise<{ valid: boolean }> {
    return apiFetch(API_BASE, '/api/auth/verify-code', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};

export default AuthAPI;
